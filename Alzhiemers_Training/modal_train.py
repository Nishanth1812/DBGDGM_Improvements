import modal
from pathlib import Path

# Define the Modal App
app = modal.App("dbgdgm-alzheimers-training")

# The training directory (just the code, no data folder)
training_dir = Path(__file__).parent

# Define the image with all required dependencies.
# We add ONLY the source Python files -- not the data/ folder.
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1")
    .pip_install(
        "torch",
        "torch-geometric",
        "networkx",
        "nilearn",
        "scikit-learn",
        "numpy",
        "scipy",
        "opencv-python-headless"
    )
    .add_local_file(training_dir / "main.py", remote_path="/root/train/main.py")
    .add_local_dir(training_dir / "src", remote_path="/root/train/src")
)

# Persistent Volume for data (uploaded) and results/checkpoints (outputs)
vol = modal.Volume.from_name("alzheimers-data", create_if_missing=True)


@app.function(image=image, gpu="a10g", volumes={"/vol": vol}, timeout=86400)
def train_on_modal(dataset: str = "oasis", categorical_dim: int = 3, trial: int = 1,
                   window_size: int = 15, window_stride: int = 5, grid_size: int = 15):

    import sys
    import os
    import shutil

    # Add code to path
    sys.path.insert(0, "/root/train")
    os.chdir("/root/train")

    # Symlink /vol/data → /root/train/data so main.py finds it at
    # Path(__file__).parent / "data"
    vol_data = Path("/vol/data")
    local_data = Path("/root/train/data")

    if not vol_data.exists() or not any(vol_data.iterdir()):
        raise RuntimeError(
            "No data found in /vol/data! "
            "Please run upload_data.py and extract_data.py first."
        )

    if local_data.exists() and not local_data.is_symlink():
        shutil.rmtree(local_data)
    if not local_data.exists():
        local_data.symlink_to(vol_data)
        print(f"Symlinked {local_data} -> {vol_data}")

    # Redirect all outputs (logs, checkpoints) to the persistent volume
    output_dir = Path(f"/vol/runs/trial_{trial}_{dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    print(f"\n{'='*50}")
    print(f"Starting Alzheimer's (OASIS) Modal training:")
    print(f"  Trial      : {trial}")
    print(f"  Dataset    : {dataset}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*50}\n")

    from main import main

    class Args:
        pass

    args = Args()
    args.dataset       = dataset
    args.categorical_dim = categorical_dim
    args.trial         = trial
    args.gpu           = 0
    args.valid_prop    = 0.1
    args.test_prop     = 0.1
    args.command       = "train"
    args.window_size   = window_size
    args.window_stride = window_stride
    args.grid_size     = grid_size
    args.resume_from   = None

    # Pass vol.commit so every checkpoint is immediately flushed to the volume.
    # This means it's safe to stop the app at any point — the last saved
    # checkpoint_best_valid.pt will always be intact on the volume.
    import functools
    from src.train import train as _train_fn
    _orig_train = _train_fn

    def _train_with_commit(model, dataset, **kwargs):
        kwargs['on_checkpoint'] = lambda: vol.commit()
        return _orig_train(model, dataset, **kwargs)

    import src.train as _train_mod
    _train_mod.train = _train_with_commit

    main(args)

    # Persist all outputs
    vol.commit()
    print(f"\nTraining complete! Download results with:")
    print(f"  python -m modal volume get alzheimers-data /runs/trial_{trial}_{dataset} ./local_results/")


@app.local_entrypoint()
def run_modal_training(dataset: str = "oasis", categorical_dim: int = 3, trial: int = 1,
                       window_size: int = 15, window_stride: int = 5, grid_size: int = 15):

    # .remote() blocks until the remote function finishes (training + vol.commit()).
    # Once it returns, training is done and results are saved to the volume.
    train_on_modal.remote(dataset, categorical_dim, trial, window_size, window_stride, grid_size)

    print("\nTraining finished and results committed to volume. Stopping Modal app...")
    # Explicitly stop the app so the container is torn down immediately.
    app.stop()
