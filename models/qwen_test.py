from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

MODEL = os.getenv("QWEN_MODEL_ID", "QWEN/qwen-3.5-small")


def main():
    print("Loading tokenizer & model:", MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
    )

    prompt = (
        "You are an ICS asset mapping assistant. "
        "List three sensors in a small water-treatment tank."
    )
    out = pipe(prompt)[0]["generated_text"]
    print("OUTPUT:\n", out)


if __name__ == "__main__":
    main()
