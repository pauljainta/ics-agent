from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

MODEL = os.getenv("QWEN_MODEL_ID", "QWEN/qwen-3.5-small")


def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        device_map="auto",
    )
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )
    return HuggingFacePipeline(pipeline=hf_pipe)


def main():
    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=[],
        template=(
            "You are an ICS scenario writer. "
            "Generate a single-step scenario inject about a sensor spoofing."
        ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run({}))


if __name__ == "__main__":
    main()
