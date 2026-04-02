import modal
import os

# 1. Define the Modal Image
# We include all necessary dependencies for transformers, peft, and gradio.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "peft",
        "torch",
        "gradio",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "bitsandbytes"
    )
)

app = modal.App("code-optimizer-app", image=image)

# 2. Define the Model Class
@app.cls(
    gpu="A100",  # Requesting an A100 GPU
    timeout=600, # 10 minute timeout for long generations
    scaledown_window=300, # Use scaledown_window instead of container_idle_timeout
)
class Model:
    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch

        print("🚀 Loading model and adapter onto A100...")
        base_model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
        adapter_repo_id = "SeifElden2342532/Code-Optimizer"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        # Load base model in bfloat16 for A100 efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load and merge the LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, adapter_repo_id)
        self.model = self.model.merge_and_unload()
        print("✅ Model loaded successfully!")

    @modal.method()
    def generate(self, code: str, category: str):
        import torch
        
        messages = [
            {
                "role": "system", 
                "content": "You are an expert Python code optimizer. Your goal is to take user-provided Python code and optimize it for performance, readability, or conciseness, based on the user's specified category. Provide the optimized code, a brief explanation of the changes, and a complexity comparison table (e.g., time and space complexity before and after optimization)."
            },
            {
                "role": "user", 
                "content": f"Original Code:\n```python\n{code}\n```\nCategory: {category}"
            }
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=1024)
        
        # Trim the input from the output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

# 3. Define the Gradio Web Interface
@app.function()
@modal.web_server(8000)
@modal.concurrent(max_inputs=10) # Correct keyword argument is 'max_inputs'
def web():
    import gradio as gr

    def optimize_ui(code, category):
        # Call the model class method
        return Model().generate.remote(code, category)

    with gr.Blocks(title="Python Code Optimizer (Modal A100)") as demo:
        gr.Markdown("# 🚀 Python Code Optimizer on Modal A100")
        gr.Markdown("Fine-tuned Qwen2.5-Coder-7B-Instruct running on high-performance hardware.")
        
        with gr.Row():
            with gr.Column():
                code_input = gr.Textbox(
                    label="Original Python Code", 
                    placeholder="Paste your Python code here...", 
                    lines=10
                )
                category_input = gr.Dropdown(
                    choices=["Performance", "Readability", "Conciseness"], 
                    value="Performance", 
                    label="Optimization Category"
                )
                submit_btn = gr.Button("Optimize", variant="primary")
            
            with gr.Column():
                output_display = gr.Markdown(label="Optimized Result")

        submit_btn.click(
            fn=optimize_ui, 
            inputs=[code_input, category_input], 
            outputs=output_display
        )
        
        gr.Examples(
            examples=[
                ["def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)", "Performance"],
                ["x = [i for i in range(10) if i % 2 == 0]", "Readability"]
            ],
            inputs=[code_input, category_input]
        )

    # Explicitly launch Gradio with server_name and server_port for Modal compatibility
    demo.launch(server_name="0.0.0.0", server_port=8000)
