import os
import sys

import modal

app = modal.App("code-opt-finetune")

volume = modal.Volume.from_name("code-opt-vol", create_if_missing=True)

VOLUME_MOUNT = "/checkpoints"

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "peft==0.13.2",
        "trl==0.12.1",
        "bitsandbytes==0.44.1",
        "huggingface_hub",
        "scipy",
    )
    .add_local_dir(".", remote_path="/root/project")
)

app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "peft==0.13.2",
        "trl==0.12.1",
        "bitsandbytes==0.44.1",
        "huggingface_hub",
        "scipy",
        "streamlit",
    )
    .add_local_dir(".", remote_path="/root/project")
)

HF_SECRET = modal.Secret.from_name("huggingface-secret")


def _setup():
    sys.path.insert(0, "/root/project")
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


@app.function(
    image=base_image,
    gpu="H100",
    timeout=60 * 60 * 10,
    volumes={VOLUME_MOUNT: volume},
    secrets=[HF_SECRET],
)
def train_remote():
    _setup()
    from src.training import run_training
    run_training()
    volume.commit()


@app.function(
    image=app_image,
    gpu="T4",
    timeout=60 * 60 * 2,
    volumes={VOLUME_MOUNT: volume},
    secrets=[HF_SECRET],
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8501)
def serve_app():
    import subprocess
    subprocess.Popen(
        [
            "streamlit", "run", "/root/project/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
        ],
    )


@app.function(image=base_image, volumes={VOLUME_MOUNT: volume})
def _list_files() -> list[tuple[str, int]]:
    results = []
    for root, _, files in os.walk(VOLUME_MOUNT):
        for f in files:
            full = os.path.join(root, f)
            results.append((full, os.path.getsize(full)))
    return results


@app.function(image=base_image, volumes={VOLUME_MOUNT: volume})
def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@app.local_entrypoint()
def main(mode: str = "train"):
    if mode == "train":
        train_remote.remote()
    elif mode == "download":
        out_dir = "./outputs"
        os.makedirs(out_dir, exist_ok=True)

        files = _list_files.remote()
        if not files:
            print("Volume is empty.")
            return

        for remote_path, size in files:
            print(f"{remote_path}  ({size / 1e6:.1f} MB)")
            rel   = os.path.relpath(remote_path, VOLUME_MOUNT)
            local = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            data  = _read_file.remote(remote_path)
            with open(local, "wb") as fh:
                fh.write(data)
            print(f"  -> {local}")
    else:
        print(f"Unknown mode '{mode}'. Use: train | download")