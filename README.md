# Code Optimization Fine-tuning

Fine-tunes `Qwen2.5-Coder-7B-Instruct` on the `SeifElden2342532/Code-Optimization` dataset using QLoRA on a Modal H100.

## Structure

```
code_opt_finetune/
├── train_modal.py
├── app.py
├── config.py
└── src/
    ├── data/
    │   └── dataset.py
    └── training/
        └── trainer.py
```

## Setup

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with your Modal token

```bash
modal token set --token-id <ID> --token-secret <SECRET>
```

### 3. Create a Modal secret for your Hugging Face token

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

## Run training

```bash
modal run train_modal.py
```

## Download results

```bash
modal run train_modal.py -- --mode download
```

Results are saved to `./outputs/`.

## Run the Streamlit app

Install dependencies locally:

```bash
modal serve train_modal.py
```


The sidebar lets you point to the adapter path (defaults to `./outputs/code-opt-qwen`).
Paste any Python code on the left, click **Optimize**, and the model output appears on the right.

## Config

Edit `config.py` before training:

| Key | Default | Description |
|---|---|---|
| `model_id` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Base model |
| `num_train_epochs` | `3` | Training epochs |
| `lora_r` | `64` | LoRA rank |
| `lora_alpha` | `128` | LoRA alpha |
| `max_seq_length` | `2048` | Max token length |
| `hf_output_repo` | `""` | Set to push adapter to HF Hub |

## Dataset format

Each example is formatted as a 3-turn chat:
- **system**: code optimization expert persona
- **user**: original code + category
- **assistant**: optimized code + explanation + complexity table
