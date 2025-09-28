# reinforce_pg_qwen_kl.py — REINFORCE + KL/Entropy + ChatTemplate + 安全サンプラ
import os, random, torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch.nn.functional as F

# ====== 環境 ======
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); random.seed(42)
EPS = 1e-8

# ====== 数値ユーティリティ ======
def safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.float()
    logits = torch.clamp(logits, -50.0, 50.0)
    logits = logits - logits.amax(dim=dim, keepdim=True)
    probs = F.softmax(logits, dim=dim)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=EPS)
    probs = probs / probs.sum(dim=dim, keepdim=True)
    return probs

def nucleus_pick_from_probs(probs_1d: torch.Tensor, top_p: float = 0.9) -> int:
    p = probs_1d.float()
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    s = p.sum()
    if not torch.isfinite(s) or s <= 0:
        V = p.numel(); return int(torch.randint(0, V, (1,), device=p.device).item())
    p = p / s
    sorted_p, sorted_idx = torch.sort(p, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=-1)
    keep = cumsum <= top_p
    if keep.numel() == 0: keep = torch.zeros_like(sorted_p, dtype=torch.bool); keep[0] = True
    keep[0] = True
    filt_idx, filt_p = sorted_idx[keep], sorted_p[keep]
    s2 = filt_p.sum()
    if not torch.isfinite(s2) or s2 <= 0:
        V = p.numel(); return int(torch.randint(0, V, (1,), device=p.device).item())
    filt_p = filt_p / s2
    return int(filt_idx[torch.multinomial(filt_p, 1)].item())

def last_logits(model: AutoModelForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[:, -1, :].float()

# ====== プロンプト ======
prompts = [
    "I really love this!",
    "This was amazing, wonderful and fantastic.",
    "The movie made me so happy.",
    "What a great experience!",
]

# ====== モデル読み込み（FP32で堅牢化） ======
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"

def encode_chat(prompt: str):
    # Instruct系はchatテンプレート必須
    msgs = [{"role":"user","content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True)
    return enc.input_ids.to(device), enc.attention_mask.to(device)

policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
policy.train()

# 参照（固定）モデル：初期重みで凍結
ref_policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
ref_policy.eval()
for p in ref_policy.parameters(): p.requires_grad_(False)

# ====== 終端一括報酬：応答のみ評価 ======
sentiment = pipeline("sentiment-analysis")
def reward_fn_response_only(resp_text: str) -> float:
    res = sentiment(resp_text, truncation=True, max_length=256)[0]
    return float(res["score"] if res["label"].upper().startswith("POS") else 1.0 - res["score"])

# ====== 生成＋logπ/Entropy/KL材料の収集（勾配有効） ======
def sample_with_stats(prompt: str, max_new: int = 32, temperature: float = 0.9, top_p: float = 0.9
) -> Tuple[str, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    戻り値:
      resp_text,
      logp_list:    現方策の logπ_t（requires_grad=True）
      logp_ref_list:参照方策の logπ_ref_t（no grad）
      ent_list:     現方策分布のエントロピー H_t（requires_grad=True）
    """
    input_ids, attn = encode_chat(prompt)
    eos = tok.eos_token_id

    gen_ids: List[int] = []
    logp_list, logp_ref_list, ent_list = [], [], []

    for _ in range(max_new):
        # 現方策
        logits = last_logits(policy, input_ids, attn)
        if temperature and temperature > 0: logits = logits / float(temperature)
        probs = safe_softmax(logits, dim=-1)          # (1,V)
        probs_1d = probs[0]

        # エントロピー
        ent_t = -(probs_1d * (probs_1d.add(EPS).log())).sum()  # scalar, gradあり

        # サンプル
        next_token_id = nucleus_pick_from_probs(probs_1d, top_p=top_p)

        # 現方策logp
        logp_t = torch.log(probs_1d[next_token_id] + EPS)      # scalar, gradあり
        logp_list.append(logp_t); ent_list.append(ent_t)

        # 参照方策logp（no grad）
        with torch.no_grad():
            ref_logits = last_logits(ref_policy, input_ids, attn)
            ref_probs = safe_softmax(ref_logits, dim=-1)[0]
            logp_ref_list.append(torch.log(ref_probs[next_token_id] + EPS))

        # 前進
        gen_ids.append(next_token_id)
        nxt = torch.tensor([[next_token_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, nxt], dim=1)
        attn = torch.cat([attn, torch.ones_like(nxt)], dim=1)

        if next_token_id == eos: break

    resp_text = tok.decode(gen_ids, skip_special_tokens=True)
    return resp_text, logp_list, logp_ref_list, ent_list

# ====== 学習ハイパラ ======
opt = torch.optim.AdamW(policy.parameters(), lr=5e-6)  # 少し下げる
beta_kl   = 0.02   # KL 正則化係数（必要に応じて 0.01〜0.1）
ent_coef  = 0.01   # エントロピー・ボーナス
ema_b     = 0.0    # ベースライン（EMA）
ema_alpha = 0.1

def reinforce_step(prompt: str):
    global ema_b
    opt.zero_grad(set_to_none=True)

    resp_text, logp_list, logp_ref_list, ent_list = sample_with_stats(prompt)
    R = reward_fn_response_only(resp_text)

    if len(logp_list) == 0:
        return R, resp_text

    # ベースライン更新
    ema_b = (1 - ema_alpha) * ema_b + ema_alpha * R
    A = R - ema_b  # advantage（全時刻同一スカラー）

    logp = torch.stack(logp_list).sum()                 # Σ_t logπ_t
    kl_seq = torch.stack([lp - lpr for lp, lpr in zip(logp_list, logp_ref_list)]).sum()  # 近似 KL = E_new[logπ−logπ_ref]
    ent = torch.stack(ent_list).mean()

    # 最小化する損失： L = -(A * Σlogπ) + β * KL  − λ * Entropy
    loss = -(A * logp) + beta_kl * kl_seq - ent_coef * ent
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    return R, resp_text

# ====== 可視化（安全サンプラ・テンプレート使用） ======
@torch.no_grad()
def generate_for_view(prompt: str, max_new: int = 64) -> str:
    policy.eval()
    input_ids, attn = encode_chat(prompt)
    eos = tok.eos_token_id
    out_ids: List[int] = []
    for _ in range(max_new):
        logits = last_logits(policy, input_ids, attn)
        probs = safe_softmax(logits, dim=-1)[0]
        nxt = nucleus_pick_from_probs(probs, top_p=0.9)
        out_ids.append(nxt)
        t = torch.tensor([[nxt]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, t], dim=1)
        attn = torch.cat([attn, torch.ones_like(t)], dim=1)
        if nxt == eos: break
    return tok.decode(out_ids, skip_special_tokens=True)

# ====== 実行 ======
if __name__ == "__main__":
    print("=== Before (no training) ===")
    for p in random.sample(prompts, 2):
        print("-", generate_for_view(p))

    steps = 20
    for t in range(steps):
        p = random.choice(prompts)
        R, resp = reinforce_step(p)
        if (t + 1) % 5 == 0:
            print(f"step {t+1}/{steps}  reward={R:.3f}  sample='{resp[:80]}'")

    print("\n=== After (policy updated) ===")
    for p in random.sample(prompts, 2):
        print("-", generate_for_view(p))
