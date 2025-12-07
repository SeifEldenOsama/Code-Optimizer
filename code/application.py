!pip install transformers peft accelerate streamlit pyngrok sentencepiece bitsandbytes

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %%writefile model_loader.py
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# import torch
# 
# BASE_MODEL = "mistralai/Mistral-7B-v0.1"
# LORA_PATH = "/content/drive/MyDrive/mistral-finetuned"
# 
# def load_model():
#     # 1. Define 4-bit quantization configuration
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=False,
#     )
# 
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
#     tokenizer.pad_token = tokenizer.eos_token
# 
#     # 2. Load the base model with quantization
#     base = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL,
#         quantization_config=bnb_config, # Apply 4-bit quantization
#         device_map="auto", # Use auto to distribute the quantized model
#     )
# 
#     # 3. Load the PeftModel (LoRA adapter)
#     # We can remove the offload_folder now as quantization should solve the OOM
#     model = PeftModel.from_pretrained(
#         base,
#         LORA_PATH,
#     )
# 
#     return model, tokenizer
#

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import torch
# from model_loader import load_model
# 
# # Use st.cache_resource to load the model only once
# @st.cache_resource
# def get_model_and_tokenizer():
#     return load_model()
# 
# st.title("Mistral Code Optimizer – Fine-tuned Model")
# 
# st.write("Enter your code to optimize:")
# 
# # Call the cached function
# model, tokenizer = get_model_and_tokenizer()
# 
# user_input = st.text_area("Your code / instruction", height=200)
# 
# if st.button("Generate"):
#     with st.spinner("Optimizing..."):
# 
#         prompt = f"<s>[INST] {user_input} [/INST]"
# 
#         # Ensure inputs are on the correct device (cuda)
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# 
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             temperature=0.3,
#         )
# 
#         result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # Clean up the prompt from the result
#         result = result.replace(prompt, "").strip()
# 
#         st.write("### Optimized Code:")
#         st.code(result)
#

import os
from pyngrok import ngrok
import subprocess
import time
import sys

NGROK_AUTH_TOKEN = "36117wvAoozGww5SI1qjgJDptFx_2uRizsKgKFFMaxkq6wEQT"
STREAMLIT_PORT = 8501
STREAMLIT_APP = "app.py"

print("Killing all existing ngrok tunnels...")
ngrok.kill()

print(f"Checking for and killing any process on port {STREAMLIT_PORT}...")
try:
    subprocess.run(
        ["fuser", "-k", f"{STREAMLIT_PORT}/tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False
    )
    time.sleep(1)
    print("Port cleared.")
except Exception as e:
    print(f"Could not clear port: {e}")

try:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
except Exception as e:
    print(f"Error setting ngrok authtoken: {e}")
    sys.exit(1)

print(f"Starting Streamlit app '{STREAMLIT_APP}' on port {STREAMLIT_PORT}...")
streamlit_command = [
    "streamlit", "run", STREAMLIT_APP,
    "--server.port", str(STREAMLIT_PORT),
    "--server.headless", "true"
]

streamlit_process = subprocess.Popen(
    streamlit_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
print(f"Streamlit process started with PID: {streamlit_process.pid}")

time.sleep(10)

if streamlit_process.poll() is not None:
    print("\n--- Streamlit App Failed to Start ---")
    print(f"Process exited with return code: {streamlit_process.returncode}")
    stdout, stderr = streamlit_process.communicate()
    if stdout:
        print("\n[STDOUT]")
        print(stdout)
    if stderr:
        print("\n[STDERR]")
        print(stderr)
    print("--------------------------------------\n")
    print("The Streamlit app crashed. Please check the [STDERR] output above for the cause.")
    streamlit_process.terminate()
else:
    try:
        print("Streamlit process is running. Creating ngrok tunnel...")
        public_url = ngrok.connect(STREAMLIT_PORT).public_url
        print(f"\nYour Streamlit app is publicly available at: {public_url}")
        print("The tunnel will remain open until you stop this cell execution.")

        while True:
            if streamlit_process.poll() is not None:
                print("\n--- Streamlit App CRASHED AFTER STARTUP ---")
                print("The Streamlit process terminated unexpectedly. Check the logs for details.")
                break
            time.sleep(1)

    except Exception as e:
        print(f"\nAn error occurred while setting up ngrok: {e}")
        streamlit_process.terminate()

ngrok.kill()

