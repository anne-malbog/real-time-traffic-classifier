"""Phase 2, step 1: acquire a labeled dataset from Roboflow.

Downloads a Roboflow-hosted dataset (already in YOLO format: images + one
.txt label file per image + a data.yaml) into data/raw/<dataset-slug>/.
This is the "Acquire/download the dataset" step of the dataset pipeline —
frame extraction, cleaning, and re-organization into the project's
images/{train,val,test} + labels/{train,val,test} layout happen in a
separate step (prepare_dataset.py) so each stage can be verified on its own.

Requires ROBOFLOW_API_KEY in a local .env file (see .env.example).

Usage:

    python -m training.download_dataset --workspace roboflow-gw7yv --project vehicles-openimages --version 1

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

DATA_RAW_DIR = Path("data/raw")


def download(workspace: str, project: str, version: int, fmt: str = "yolov8") -> Path:
    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not found. Copy .env.example to .env (if you haven't) "
            "and paste your private API key from app.roboflow.com/settings/api"
        )

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)

    # Note: the Roboflow SDK's Project object doesn't expose a license field
    # directly — verify/document license on the project's public page before
    # using a new dataset (see README's Dataset section for what's confirmed).
    print("=" * 60)
    print(f"Project:      {proj.name}")
    print(f"Public:       {proj.public}")
    print(f"Type:         {proj.type}")
    print(f"Classes:      {proj.classes}")
    print(f"Splits:       {proj.splits}")
    print("=" * 60)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW_DIR / project
    dataset = ver.download(fmt, location=str(dest))

    print(f"\nDownloaded to: {dataset.location}")
    return Path(dataset.location)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a Roboflow dataset in YOLO format")
    p.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    p.add_argument("--project", required=True, help="Roboflow project slug")
    p.add_argument("--version", type=int, required=True, help="Dataset version number")
    p.add_argument("--format", default="yolov8", help="Export format (default: yolov8)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(args.workspace, args.project, args.version, args.format)
