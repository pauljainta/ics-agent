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
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_DEVICE if DEFAULT_DEVICE in {"auto", "cpu", "cuda"} else "auto",
        help="Execution target. auto picks cuda when available, otherwise cpu.",
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
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested_precision: str):
    if requested_precision == "bf16":
        return torch.bfloat16
    return torch.float16


def load_model(model_ref: str, offline: bool, resolved_device: str, requested_precision: str):
    device_map = "auto" if resolved_device == "cuda" else "cpu"
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "local_files_only": offline,
        "cache_dir": CACHE_DIR,
    }

    if resolved_device == "cuda":
        if requested_precision == "4bit":
            load_kwargs["load_in_4bit"] = True
        elif requested_precision == "8bit":
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["torch_dtype"] = resolve_dtype(requested_precision)
    else:
        load_kwargs["torch_dtype"] = torch.float32

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
    try:
        model = load_model(model_ref, args.offline, resolved_device, args.precision)
    except Exception as exc:
        if resolved_device == "cuda":
            print(f"WARNING: GPU load failed ({exc}). Falling back to CPU.")
            model = load_model(model_ref, args.offline, "cpu", args.precision)
        else:
            raise

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
