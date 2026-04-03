"""FastAPI inference server for KazLLM.

Loads a trained checkpoint and serves text generation via REST API.

Usage:
    # From project root:
    PYTHONPATH=src python scripts/serve.py --checkpoint checkpoints/kaz_nano --port 8000

    # Or with a specific device:
    PYTHONPATH=src python scripts/serve.py --checkpoint checkpoints/kaz_nano --device cpu
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerFast

from kazllm.model.config import KazLLMConfig
from kazllm.model.model import KazLLMForCausalLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="KazLLM API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state (populated at startup)
_model: KazLLMForCausalLM | None = None
_tokenizer: PreTrainedTokenizerFast | None = None
_device: torch.device = torch.device("cpu")
_model_config: KazLLMConfig | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    tokens_generated: int
    tokens_per_second: float


class ModelInfo(BaseModel):
    model_name: str
    parameters: int
    architecture: str
    mtp_enabled: bool
    mhc_enabled: bool
    engram_enabled: bool
    device: str
    vocab_size: int


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    cfg = _model_config
    total_params = sum(p.numel() for p in _model.parameters())
    features = []
    if cfg.use_mhc:
        features.append(f"mHC(n={cfg.mhc_streams})")
    if cfg.use_engram:
        features.append("Engram")
    if cfg.use_mtp:
        features.append(f"MTP(D={cfg.mtp_depth})")
    arch_str = f"KazLLM-v3 {cfg.num_hidden_layers}L h={cfg.hidden_size}"
    if features:
        arch_str += " + " + " + ".join(features)
    return ModelInfo(
        model_name="KazLLM-Nano" if cfg.hidden_size <= 512 else "KazLLM-500M",
        parameters=total_params,
        architecture=arch_str,
        mtp_enabled=cfg.use_mtp,
        mhc_enabled=cfg.use_mhc,
        engram_enabled=cfg.use_engram,
        device=str(_device),
        vocab_size=cfg.vocab_size,
    )


@torch.inference_mode()
def _generate(
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> list[int]:
    """Simple autoregressive generation with top-k/top-p sampling."""
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Truncate to max context
        ctx = generated[:, -_model_config.max_position_embeddings :]
        outputs = _model(ctx)
        logits = outputs.logits[:, -1, :]  # (1, V)

        # Repetition penalty
        if repetition_penalty > 1.0:
            for token_id in set(generated[0].tolist()):
                logits[0, token_id] /= repetition_penalty

        # Temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == _tokenizer.eos_token_id:
                break
            continue

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == _tokenizer.eos_token_id:
            break

    return generated[0, input_ids.shape[1] :].tolist()


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    input_ids = _tokenizer.encode(req.prompt, return_tensors="pt").to(_device)

    t0 = time.perf_counter()
    new_tokens = _generate(
        input_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    elapsed = time.perf_counter() - t0

    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    tps = len(new_tokens) / elapsed if elapsed > 0 else 0

    return GenerateResponse(
        text=text,
        prompt=req.prompt,
        tokens_generated=len(new_tokens),
        tokens_per_second=round(tps, 1),
    )


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Server-Sent Events streaming generation."""
    import json

    input_ids = _tokenizer.encode(req.prompt, return_tensors="pt").to(_device)

    async def event_stream():
        generated = input_ids.clone()
        t0 = time.perf_counter()
        num_generated = 0

        for _ in range(req.max_new_tokens):
            ctx = generated[:, -_model_config.max_position_embeddings :]
            with torch.inference_mode():
                outputs = _model(ctx)
            logits = outputs.logits[:, -1, :] / max(req.temperature, 0.01)

            if req.top_k > 0:
                indices_to_remove = logits < torch.topk(logits, req.top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            num_generated += 1

            token_text = _tokenizer.decode([next_token.item()], skip_special_tokens=False)
            if next_token.item() == _tokenizer.eos_token_id:
                break

            data = json.dumps({"token": token_text, "done": False})
            yield f"data: {data}\n\n"

        elapsed = time.perf_counter() - t0
        tps = num_generated / elapsed if elapsed > 0 else 0
        done_data = json.dumps(
            {
                "token": "",
                "done": True,
                "tokens_generated": num_generated,
                "tokens_per_second": round(tps, 1),
            }
        )
        yield f"data: {done_data}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def load_model(checkpoint_dir: str, device: str = "auto"):
    """Load model + tokenizer from a training checkpoint directory."""
    global _model, _tokenizer, _device, _model_config

    checkpoint_path = Path(checkpoint_dir)
    tokenizer_path = Path("data/tokenizer/kaz_sp_unigram_50k/hf_tokenizer")

    # Resolve device
    if device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)
    log.info(f"Using device: {_device}")

    # Load tokenizer
    log.info(f"Loading tokenizer from {tokenizer_path}")
    _tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    # Find the latest checkpoint
    ckpt_dirs = sorted(
        checkpoint_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if ckpt_dirs:
        ckpt = ckpt_dirs[-1]
        log.info(f"Loading from checkpoint: {ckpt}")
    else:
        ckpt = checkpoint_path
        log.info(f"Loading from directory: {ckpt}")

    # Load config from the model config yaml or from checkpoint
    config_file = ckpt / "config.json"
    if config_file.exists():
        _model_config = KazLLMConfig.from_pretrained(str(ckpt))
    else:
        # Fallback: try to find the yaml config
        log.warning("No config.json in checkpoint. Attempting to load from kaz_nano.yaml")
        with open("configs/model/kaz_nano.yaml") as f:
            cfg = yaml.safe_load(f)
        _model_config = KazLLMConfig(**cfg)

    # Load model
    model_bin = ckpt / "model.safetensors"
    if not model_bin.exists():
        model_bin = ckpt / "pytorch_model.bin"

    _model = KazLLMForCausalLM(_model_config)
    if model_bin.exists():
        state_dict = torch.load(str(model_bin), map_location="cpu", weights_only=True)
        # Strip MTP modules for inference (they're training-only)
        state_dict = {k: v for k, v in state_dict.items() if "mtp_modules" not in k}
        _model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded weights from {model_bin}")
    else:
        log.warning(f"No model weights found at {ckpt}. Using random initialization!")

    _model.to(_device)
    _model.eval()

    total_params = sum(p.numel() for p in _model.parameters())
    log.info(f"Model loaded: {total_params / 1e6:.1f}M params on {_device}")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="KazLLM API Server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/kaz_nano")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.checkpoint, args.device)
    uvicorn.run(app, host=args.host, port=args.port)
