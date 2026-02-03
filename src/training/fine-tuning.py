import os
import torch
import pandas as pd
import random
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Configuration
DATA_PATH = os.getenv("DATA_PATH", "data/code_optimization_dataset.csv")
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./models/mistral-finetuned")

def prepare_dataset(path):
    df = pd.read_csv(path)
    required_cols = {"unoptimized_code", "optimized_code"}
    assert required_cols.issubset(df.columns), f"Dataset must contain columns: {required_cols}"

    def make_pair(row):
        meta = []
        if 'optimization_type' in row and pd.notna(row['optimization_type']):
            meta.append(f"Type: {row['optimization_type']}")
        if 'complexity_change' in row and pd.notna(row['complexity_change']):
            meta.append(f"Complexity: {row['complexity_change']}")
        meta_str = " | ".join(meta)
        
        instruction = f"Optimize the following Python function/code for performance and clarity.\n\n"
        if meta_str:
            instruction += f"# {meta_str}\n\n"
        instruction += "### Unoptimized code:\n" + row['unoptimized_code'].strip()
        response = row['optimized_code'].strip()
        return {"instruction": instruction, "response": response}

    pairs = [make_pair(r) for _, r in df.iterrows()]
    random.shuffle(pairs)
    

    split = int(len(pairs) * 0.9)
    train_ds = Dataset.from_list(pairs[:split])
    valid_ds = Dataset.from_list(pairs[split:])
    
    dataset = DatasetDict({"train": train_ds, "validation": valid_ds})
    
    def format_for_sft(example):
        return {"text": f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>"}

    return dataset.map(format_for_sft, remove_columns=dataset["train"].column_names)

def train():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    dataset = prepare_dataset(DATA_PATH)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    os.environ["WANDB_DISABLED"] = "true"
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
