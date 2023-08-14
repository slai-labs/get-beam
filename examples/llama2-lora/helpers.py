import os
from pathlib import Path

# The model that will be fined-tuned
base_model = "openlm-research/open_llama_7b"

# The Beam Volume
beam_volume_path = "./checkpoints"


def get_trained_model(checkpoint):
    return Path(f"{beam_volume_path}/{checkpoint}")


def get_newest_checkpoint():
    # Find all checkpoint dirs
    checkpoint_dirs = [
        d
        for d in os.listdir(beam_volume_path)
        if os.path.isdir(os.path.join(beam_volume_path, d)) and "checkpoint" in d
    ]

    if not checkpoint_dirs:
        print("No checkpoints exist yet, make sure you've trained a model.")
        return

    # Get the latest checkpoint
    most_recent_dir = max(
        checkpoint_dirs,
        key=lambda d: os.path.getctime(os.path.join(beam_volume_path, d)),
    )

    newest_checkpoint_path = os.path.join(beam_volume_path, most_recent_dir)

    return Path(newest_checkpoint_path).as_posix()
