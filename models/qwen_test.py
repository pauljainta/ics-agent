import os
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "").strip()
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "").strip() or None
DEFAULT_DEVICE = os.getenv("QWEN_DEVICE", "auto").strip().lower()
DEFAULT_PRECISION = os.getenv("QWEN_PRECISION", "auto").strip().lower()


def parse_args():
    parser = ArgumentParser(description="Minimal Qwen generation smoke test.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require local files only (no network calls).",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "You are an ICS asset mapping assistant. "
            "List three sensors in a small water-treatment tank."
        ),
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda"],
        default=DEFAULT_DEVICE if DEFAULT_DEVICE in {"auto", "cuda"} else "auto",
        help="Execution target. auto requires a CUDA-capable GPU.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16", "8bit", "4bit"],
        default=(
            DEFAULT_PRECISION
            if DEFAULT_PRECISION in {"auto", "fp16", "bf16", "8bit", "4bit"}
            else "auto"
        ),
        help="Model loading precision/quantization.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Qwen smoke test requires a CUDA-capable GPU")
    if requested == "auto" and not torch.cuda.is_available():
        raise RuntimeError("Qwen smoke test requires a CUDA-capable GPU")
    return "cuda"


def resolve_dtype(requested_precision: str):
    if requested_precision == "bf16":
        return torch.bfloat16
    return torch.float16


def load_model(
    model_ref: str, offline: bool, resolved_device: str, requested_precision: str
):
    if resolved_device != "cuda":
        raise RuntimeError("Agentic model loading requires CUDA")
    device_map = "auto"
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "local_files_only": offline,
        "cache_dir": CACHE_DIR,
    }

    if requested_precision == "4bit":
        load_kwargs["load_in_4bit"] = True
    elif requested_precision == "8bit":
        load_kwargs["load_in_8bit"] = True
    elif (
        requested_precision == "bf16"
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    ):
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = resolve_dtype(requested_precision)

    return AutoModelForCausalLM.from_pretrained(
        model_ref,
        **load_kwargs,
    )


def main():
    args = parse_args()
    model_ref = MODEL_PATH or MODEL

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("Loading tokenizer & model:", model_ref)
    print("Offline mode:", args.offline)
    if args.offline and not MODEL_PATH:
        print(
            "WARNING: --offline is enabled and QWEN_MODEL_PATH is empty. "
            "The model must already exist in local Hugging Face cache."
        )

    resolved_device = resolve_device(args.device)
    print("Requested device:", args.device)
    print("Resolved device:", resolved_device)
    print("Requested precision:", args.precision)

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=True,
        local_files_only=args.offline,
        cache_dir=CACHE_DIR,
    )
    model = load_model(model_ref, args.offline, resolved_device, args.precision)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    out = pipe(args.prompt)[0]["generated_text"]
    print("OUTPUT:\n", out)


if __name__ == "__main__":
    main()
