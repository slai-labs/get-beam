import os
from pathlib import Path

# The model that will be fined-tuned
base_model = "openlm-research/open_llama_7b"

# The Beam Volume
beam_volume_path = "./checkpoints"


def get_trained_model(checkpoint):
    return Path(f"{beam_volume_path}/{checkpoint}")


def get_newest_checkpoint():
    try:
        checkpoint_path = Path(beam_volume_path)
        checkpoint_files = checkpoint_path.glob("*.json")
        newest_checkpoint = max(checkpoint_files, key=os.path.getctime, default=None)
    except Exception as e:
        print(f"Error: {e}")

    return Path(newest_checkpoint).as_posix()
