import sys
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are an expert Python developer. "
    "Given unoptimized Python code, produce the optimized version along with a concise explanation "
    "of every optimization applied, including any changes to time or space complexity."
)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_ADAPTER_PATH = "./outputs/code-opt-qwen"

st.set_page_config(
    page_title="Python Code Optimizer",
    page_icon="⚡",
    layout="wide",
)

st.title("Python Code Optimizer")
st.caption("Powered by Qwen2.5-Coder fine-tuned on code optimization examples")


with st.sidebar:
    st.header("Model settings")
    model_id = st.text_input("Base model", value=DEFAULT_MODEL_ID)
    adapter_path = st.text_input("Adapter path (LoRA)", value=DEFAULT_ADAPTER_PATH)
    max_new_tokens = st.slider("Max new tokens", 256, 2048, 1024, 64)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.2, 0.05)
    load_button = st.button("Load model", use_container_width=True)


@st.cache_resource(show_spinner=False)
def load_model(base_model_id: str, adapter: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.isdir(adapter):
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.eval()
    return tokenizer, model


if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if load_button:
    with st.spinner("Loading model and adapter..."):
        try:
            tokenizer, model = load_model(model_id, adapter_path)
            st.session_state.model_loaded = True
            st.sidebar.success("Model loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")

st.divider()

col_in, col_out = st.columns(2)

with col_in:
    st.subheader("Original code")
    code_input = st.text_area(
        label="Paste your Python code here",
        height=400,
        placeholder="scores = {}\nfor x in range(100):\n    if x not in scores:\n        scores[x] = 0\n    scores[x] += x",
        label_visibility="collapsed",
    )
    optimize_button = st.button("Optimize", type="primary", use_container_width=True)

with col_out:
    st.subheader("Optimized output")
    output_placeholder = st.empty()

if optimize_button:
    if not st.session_state.model_loaded:
        st.warning("Load the model first using the sidebar.")
    elif not code_input.strip():
        st.warning("Paste some code to optimize.")
    else:
        tokenizer, model = load_model(model_id, adapter_path)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Optimize the following Python code.\n\n```python\n{code_input.strip()}\n```",
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with st.spinner("Generating..."):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        with output_placeholder.container():
            st.markdown(generated)
