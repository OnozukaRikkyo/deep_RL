# ppo_clip_qwen_stable.py — Qwen2.5-0.5B-Instruct 最小PPO（省メモリ対応：QLoRA + 単一モデル参照）
import os, random, re, torch
from typing import List, Tuple, Optional
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ========= メモリ設定（断片化対策） =========
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

# ========= 端末・乱数 =========
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); random.seed(42)
EPS = 1e-8

# ========= 長さ・バッチ上限（OOM回避の要） =========
MAX_CTX = 256   # 入力コンテキストの最大トークン数
MAX_NEW = 16    # 生成長
BATCH_SIZE = 1  # まずは1で安定化
PPO_EPOCHS = 2

# ========= 安定化ユーティリティ =========
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
    if keep.numel() == 0:
        keep = torch.zeros_like(sorted_p, dtype=torch.bool)
    keep[0] = True
    filt_idx, filt_p = sorted_idx[keep], sorted_p[keep]
    s2 = filt_p.sum()
    if not torch.isfinite(s2) or s2 <= 0:
        V = p.numel(); return int(torch.randint(0, V, (1,), device=p.device).item())
    filt_p = filt_p / s2
    return int(filt_idx[torch.multinomial(filt_p, 1)].item())

def last_logits_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[:, -1, :].float()  # (1,V)

# ========= データ =========
prompts = [
    "Please write a short, positive one-sentence reaction to this: I really love this!",
    "Give a cheerful one-sentence comment: This was amazing, wonderful and fantastic.",
    "One-line positive thought: The movie made me so happy.",
    "Write a short upbeat line: What a great experience!",
]

# ========= モデル（QLoRA: 4bit量子化 + LoRA, 単一モデルで参照も賄う） =========
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL)
# pad は eos を使い、語彙を増やさない
if tok.pad_token is None and tok.eos_token is not None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

def encode_chat(user_prompt: str):
    msgs = [{"role":"user","content": user_prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_CTX)
    return enc.input_ids.to(device), enc.attention_mask.to(device)

# ---- QLoRA 読み込み ----
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_cfg,
    device_map="auto",
)
# 学習安定化
base.config.use_cache = False
base.gradient_checkpointing_enable()  # 勾配チェックポイントON

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
policy = get_peft_model(base, lora_cfg)
policy.print_trainable_parameters()
policy.train()

# ★ 勾配チェックポイントと併用時の必須：入力にも勾配を許可
try:
    policy.enable_input_require_grads()
except AttributeError:
    bm = getattr(policy, "base_model", None)
    if bm is not None and hasattr(bm, "enable_input_require_grads"):
        bm.enable_input_require_grads()

# ★ 語彙サイズずれを吸収（増分は EOS 埋め込みで初期化）
def _ensure_vocab_resized(model, tokenizer):
    emb = model.get_input_embeddings().weight
    old_vocab = emb.size(0)
    new_vocab = len(tokenizer)
    if old_vocab != new_vocab:
        model.resize_token_embeddings(new_vocab)
        with torch.no_grad():
            emb2 = model.get_input_embeddings().weight
            if new_vocab > old_vocab and tokenizer.eos_token_id is not None:
                emb2[old_vocab:new_vocab, :].copy_(emb2[tokenizer.eos_token_id, :])

_ensure_vocab_resized(policy, tok)

# ---- 参照(log π_ref)は LoRAを無効化して同一モデルから取得（2体目不要） ----
def logits_ref(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    assert isinstance(policy, PeftModel)
    with policy.disable_adapter():
        out = policy.base_model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits

# ========= 報酬（応答のみ + アンチディスクレーマ） =========
sentiment = pipeline("sentiment-analysis")  # CPU推奨（GPUが逼迫する場合）

DISCLAIMER_PATTERNS = [
    r"\bAs an AI language model\b",
    r"\bI am an AI\b",
    r"\bI don't have (?:personal )?(?:experiences|emotions)\b",
]
def penalty_disclaimer(text: str) -> float:
    for pat in DISCLAIMER_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return 0.3
    return 0.0

def reward_response_only(resp_text: str) -> float:
    base_res = sentiment(resp_text, truncation=True, max_length=256)[0]
    pos = float(base_res["score"] if base_res["label"].upper().startswith("POS") else 1.0 - base_res["score"])
    pen = penalty_disclaimer(resp_text)
    return max(0.0, min(1.0, pos - pen))

# ========= Rollout（new/old logp 収集はno_gradで軽量） =========
@torch.no_grad()
def rollout_one(prompt: str, max_new: int = MAX_NEW, temperature: float = 0.7, top_p: float = 0.9):
    policy.eval()
    input_ids, attn = encode_chat(prompt)
    eos = tok.eos_token_id

    acts: List[int] = []
    old_logps: List[torch.Tensor] = []

    for _ in range(max_new):
        # new分布（LoRA有効）
        logits_new = last_logits_forward(policy, input_ids, attn)
        if temperature and temperature > 0: logits_new = logits_new / float(temperature)
        probs_new = safe_softmax(logits_new, dim=-1)[0]
        a = nucleus_pick_from_probs(probs_new, top_p=top_p)

        # old/ref分布（LoRA無効）
        logits_o = logits_ref(input_ids, attn)[:, -1, :].float()
        probs_o = safe_softmax(logits_o, dim=-1)[0]
        old_logps.append(torch.log(probs_o[a] + EPS))

        acts.append(a)
        t = torch.tensor([[a]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, t], dim=1)
        attn = torch.cat([attn, torch.ones_like(t)], dim=1)
        if a == eos: break

    resp = tok.decode(acts, skip_special_tokens=True)
    return resp, acts, (torch.stack(old_logps) if len(old_logps) else torch.empty(0, device=device))

# ========= 勾配ONの再計算（teacher forcing） =========
def recompute_new_logps_and_entropy(prompt: str, actions: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    # 親スコープがno_gradでも確実に勾配を有効化
    with torch.enable_grad():
        policy.train()
        input_ids, attn = encode_chat(prompt)
        logps, ents = [], []
        for a in actions:
            logits = last_logits_forward(policy, input_ids, attn)  # 勾配が通る経路
            probs = safe_softmax(logits, dim=-1)[0]
            lp = torch.log(probs[a] + EPS)
            ent = -(probs * (probs.add(EPS).log())).sum()

            if not lp.requires_grad:
                raise RuntimeError("new_logp does not require grad — check enable_input_require_grads & no_grad scopes")

            logps.append(lp); ents.append(ent)
            t = torch.tensor([[a]], device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, t], dim=1)
            attn = torch.cat([attn, torch.ones_like(t)], dim=1)
        return torch.stack(logps), torch.stack(ents)

# ========= PPO更新 =========
# OptimizerはbitsandbytesのPagedAdamW32bitを優先（失敗時は標準AdamW）
try:
    from bitsandbytes.optim import PagedAdamW32bit as OptimClass
except Exception:
    OptimClass = torch.optim.AdamW

optimizer = OptimClass(
    policy.parameters(),
    lr=5e-6, betas=(0.9, 0.999), weight_decay=0.0
)

clip_eps = 0.2
beta_kl = 0.05
ent_coef = 0.02
ema_b = 0.0
ema_alpha = 0.1

def collect_batch():
    batch = []
    for _ in range(BATCH_SIZE):
        prompt = random.choice(prompts)
        resp, acts, old_logp = rollout_one(prompt, max_new=MAX_NEW)
        R = reward_response_only(resp)
        batch.append({"prompt": prompt, "resp": resp, "acts": acts, "old_logp": old_logp, "R": R})
    return batch

def ppo_update(batch):
    global ema_b
    if not batch:
        return
    # EMA baseline
    mean_R = sum(b["R"] for b in batch) / len(batch)
    ema_b = (1 - ema_alpha) * ema_b + ema_alpha * mean_R

    for _ in range(PPO_EPOCHS):
        losses = []
        for b in batch:
            acts = b["acts"]
            if not acts:
                continue
            new_logp, ent = recompute_new_logps_and_entropy(b["prompt"], acts)  # (T,), (T,)
            old_logp = b["old_logp"].detach() if b["old_logp"].numel() > 0 else torch.zeros_like(new_logp)
            ratio = torch.exp(new_logp - old_logp)  # (T,)

            # Advantage（全時刻同一）＋標準化でスケール安定化
            A = b["R"] - ema_b
            A_vec = torch.full_like(new_logp, float(A))
            A_std = A_vec.std().clamp_min(1e-6)
            A_vec = (A_vec - A_vec.mean()) / A_std

            pg1 = ratio * A_vec
            pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A_vec
            policy_loss = -torch.min(pg1, pg2).mean()

            # 近似KL: E_new[logπ_new − logπ_ref] ≈ (new_logp − old_logp).mean()
            kl_est = (new_logp - old_logp).mean()
            entropy = ent.mean()

            loss = policy_loss + beta_kl * kl_est - ent_coef * entropy
            losses.append(loss)

        if not losses:
            continue
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.stack(losses).mean()  # ここでの .float() は不要

        # ★ 本当にグラフが繋がっているか確認（開発時は残すと安心）
        if not total_loss.requires_grad:
            raise RuntimeError("total_loss is not attached to a graph. Check new_logp path and no_grad scopes.")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # テンソル掃除（量子化モデルで効果的）
        del total_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# ========= 可視化生成（安全サンプラ・チャットテンプレ） =========
@torch.no_grad()
def generate_for_view(prompt: str, max_new: int = MAX_NEW) -> str:
    policy.eval()
    input_ids, attn = encode_chat(prompt)
    eos = tok.eos_token_id
    out_ids: List[int] = []
    for _ in range(max_new):
        logits = last_logits_forward(policy, input_ids, attn)
        probs = safe_softmax(logits, dim=-1)[0]
        nxt = nucleus_pick_from_probs(probs, top_p=0.9)
        out_ids.append(nxt)
        t = torch.tensor([[nxt]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, t], dim=1)
        attn = torch.cat([attn, torch.ones_like(t)], dim=1)
        if nxt == eos: break
    return tok.decode(out_ids, skip_special_tokens=True)

# ========= 実行 =========
if __name__ == "__main__":
    print("=== Before (no training) ===")
    for p in random.sample(prompts, min(2, len(prompts))):
        print("-", generate_for_view(p))

    iters = 5
    for it in range(1, iters + 1):
        batch = collect_batch()
        avg_R = sum(b["R"] for b in batch) / len(batch) if batch else 0.0
        ppo_update(batch)
        print(f"iter {it}/{iters}  avg_reward={avg_R:.3f}")
        sample_p = random.choice(prompts)
        print(" prompt:", sample_p)
        print(" sample:", generate_for_view(sample_p), "\n")

    print("=== After (policy updated) ===")
    for p in random.sample(prompts, min(2, len(prompts))):
        print("-", generate_for_view(p))
