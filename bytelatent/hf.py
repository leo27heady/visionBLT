import os
from bytelatent.entropy_model import load_entropy_model
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.transformer import LMTransformer
import typer

from bytelatent.generate import load_consolidated_model_and_tokenizer


app = typer.Typer()


@app.command()
def convert_to_transformers(blt_weights_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(blt_weights_dir)
    blt_dir = os.path.join(output_dir, "blt")
    entropy_dir = os.path.join(output_dir, "entropy")
    model.save_pretrained(blt_dir)
    blt_readme_file = os.path.join(blt_dir, "README.md")
    if os.path.exists(blt_readme_file):
        os.remove(blt_readme_file)

    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = False
    print("Loading entropy model and patcher")
    patcher_args.entropy_model_checkpoint_dir = os.path.join(
        blt_weights_dir, "entropy_model"
    )
    patcher = patcher_args.build()
    state_path = os.path.join(
        patcher_args.entropy_model_checkpoint_dir, "consolidated.pth"
    )
    entropy_model = load_entropy_model(
        patcher_args.entropy_model_checkpoint_dir, state_path
    )
    entropy_model.save_pretrained(entropy_dir)
    entropy_readme_file = os.path.join(entropy_dir, "README.md")
    if os.path.exists(entropy_readme_file):
        os.remove(entropy_readme_file)

    # TODO: Persist tokenizer in HF compatible way


@app.command()
def load_transformers(
    source: str,
    entropy_repo: str = "facebook/blt-entropy",
    blt_repo: str = "facebook/blt-1b",
    entropy_dir: str | None = None,
    blt_dir: str | None = None,
):
    if source == "local":
        assert entropy_dir is not None
        assert blt_dir is not None
        entropy_model = LMTransformer.from_pretrained(
            entropy_dir, local_files_only=True
        )
        blt_model = ByteLatentTransformer.from_pretrained(
            blt_dir, local_files_only=True
        )
    elif source == "hub":
        entropy_model = LMTransformer.from_pretrained(entropy_repo)
        blt_model = ByteLatentTransformer.from_pretrained(blt_repo)
    else:
        raise ValueError(f"Unknown source: {source}")

    # TODO: Need a way to get tokenizer
    # TODO: Need a way to get patching settings
    # TODO: Insert test inference call


if __name__ == "__main__":
    app()
