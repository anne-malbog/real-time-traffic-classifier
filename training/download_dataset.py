"""Phase 2, Step 1: Download the dataset.

This script downloads a labeled vehicle dataset from Roboflow in YOLO format. 
The dataset is saved in data/raw/, while cleaning and organizing the data 
for training are handled in the next step.
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

    # Note: the Roboflow SDK's Project object doesn't expose a license field directly. 
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
