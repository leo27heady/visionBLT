import os

import typer
from huggingface_hub import snapshot_download


def main(models: list[str] = ["blt-1b", "blt-7b"]):
    if not os.path.exists("hf-weights"):
        os.makedirs("hf-weights")
    for model in models:
        snapshot_download(f"facebook/{model}", local_dir=f"hf-weights/{model}")


if __name__ == "__main__":
    typer.run(main)
