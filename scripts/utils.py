from pathlib import Path
import json


def ensure_directory(path: Path):
    """
    Ensure a directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_metadata(metadata: dict, output_path: Path):
    """
    Save run metadata for reproducibility and auditability.
    """
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
