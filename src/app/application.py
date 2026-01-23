import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = os.getenv("LORA_PATH", "./models/mistral-finetuned")

@st.cache_resource
def load_model():
    """Load the base model with 4-bit quantization and apply LoRA adapters."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base,
        LORA_PATH,
    )

    return model, tokenizer

def main():
    st.set_page_config(page_title="AI Code Optimizer", page_icon="🚀")
    st.title("Mistral Code Optimizer")
    st.write("Enter your Python code below to receive an optimized version.")

    model, tokenizer = load_model()

    user_input = st.text_area("Python Code", height=300, placeholder="def my_function()...")

    if st.button("Optimize Code"):
        if not user_input.strip():
            st.warning("Please enter some code first.")
            return

        with st.spinner("Analyzing and optimizing..."):
            prompt = f"<s>[INST] Optimize the following Python function/code for performance and clarity.\n\n### Unoptimized code:\n{user_input} [/INST]"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True
                )

            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Clean up the prompt from the result
            result = result.replace(prompt, "").strip()

            st.subheader("Optimized Code")
            st.code(result, language="python")

if __name__ == "__main__":
    main()
