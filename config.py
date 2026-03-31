from dataclasses import dataclass


@dataclass
class Config:
    hf_dataset_id: str = "SeifElden2342532/Code-Optimization"
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    output_dir: str = "/checkpoints/code-opt-qwen"
    hf_output_repo: str = ""

    train_split: float = 0.9
    seed: int = 42

    max_seq_length: int = 2048
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    dataloader_num_workers: int = 4

    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


config = Config()
