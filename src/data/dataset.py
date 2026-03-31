from datasets import load_dataset, DatasetDict
from config import config

SYSTEM_PROMPT = (
    "You are an expert Python developer. "
    "Given unoptimized Python code, produce the optimized version along with a concise explanation "
    "of every optimization applied, including any changes to time or space complexity."
)


def format_example(row: dict) -> dict:
    user_content = (
        f"Optimize the following Python code.\n\n"
        f"Category: {row['category']} / {row['subcategory']}\n\n"
        f"```python\n{row['original_code']}\n```"
    )

    assistant_content = (
        f"**Optimized code:**\n\n"
        f"```python\n{row['optimized_code']}\n```\n\n"
        f"**Explanation:** {row['optimization_explanation']}\n\n"
        f"| | Original | Optimized |\n"
        f"|---|---|---|\n"
        f"| Time complexity | {row['original_time_complexity']} | {row['optimized_time_complexity']} |\n"
        f"| Space complexity | {row['original_space_complexity']} | {row['optimized_space_complexity']} |"
    )

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {"messages": messages}


def load_and_split() -> DatasetDict:
    raw = load_dataset(config.hf_dataset_id, split="train")
    raw = raw.map(format_example, remove_columns=raw.column_names)
    split = raw.train_test_split(
        test_size=1.0 - config.train_split,
        seed=config.seed,
    )
    return DatasetDict({"train": split["train"], "test": split["test"]})
