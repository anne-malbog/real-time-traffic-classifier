"""Build a fair baseline dataset for comparison.

The pretrained YOLO model uses COCO classes, 
while the fine-tuned models use different project-specific classes. 
This script matches equivalent classes between the two datasets and 
creates a compatible test dataset for a fair comparison. 
Classes without a COCO equivalent are excluded from the baseline comparison 
and evaluated only with the fine-tuned model.

Known name differences between this project's datasets and COCO's own names, 
for classes that ARE the same real-world category. 
Add to this list as new datasets introduce new naming conventions.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


COCO_NAME_ALIASES = {
    "motorbike": "motorcycle",  # stage-2 dataset's naming; same class as COCO's "motorcycle"
}

TMP_DIR = Path("outputs/metrics/tmp_coco_overlap")


def build_class_mapping(our_names: list[str], coco_names: dict[int, str]) -> dict[str, str]:

    coco_name_set = {v.lower() for v in coco_names.values()}
    mapping = {}
    for name in our_names:
        lname = name.lower()
        if lname in coco_name_set:
            mapping[name] = lname
        elif COCO_NAME_ALIASES.get(lname) in coco_name_set:
            mapping[name] = COCO_NAME_ALIASES[lname]
    return mapping


def build_coco_overlap_dataset(
    coco_names: dict[int, str],
    split: str = "test",
    dataset_yaml: str = "data/dataset.yaml",
    tmp_dir: Path = TMP_DIR,
) -> str:
    with open(dataset_yaml) as f:
        our_meta = yaml.safe_load(f)
    our_names = our_meta["names"]  # our index -> our name, e.g. {0: 'Ambulance', ...}
    processed_dir = Path(our_meta["path"])

    class_map = build_class_mapping(our_names, coco_names)
    coco_name_to_id = {v: k for k, v in coco_names.items()}

    # our class id is coco class id, only for names with a real COCO match
    our_id_to_coco_id = {}
    for our_id, our_name in enumerate(our_names):
        if our_name in class_map:
            our_id_to_coco_id[our_id] = coco_name_to_id[class_map[our_name]]

    src_images = processed_dir / "images" / split
    src_labels = processed_dir / "labels" / split
    dst_images = tmp_dir / "images" / split
    dst_labels = tmp_dir / "labels" / split
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for label_path in src_labels.glob("*.txt"):
        remapped_lines = []
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                our_id = int(parts[0])
                if our_id not in our_id_to_coco_id:
                    continue  # drop classes with no COCO match
                coco_id = our_id_to_coco_id[our_id]
                remapped_lines.append(" ".join([str(coco_id), *parts[1:]]))

        (dst_labels / label_path.name).write_text("\n".join(remapped_lines))

        matches = list(src_images.glob(label_path.stem + ".*"))
        if matches:
            shutil.copy2(matches[0], dst_images / matches[0].name)

    dataset_yaml_out = {
        "path": str(tmp_dir.resolve()),
        "train": "images/test",  # unused placeholders... only `split` is evaluated
        "val": "images/test",
        "test": "images/test",
        "nc": len(coco_names),
        "names": coco_names,
    }
    out_path = tmp_dir / "dataset.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(dataset_yaml_out, f, sort_keys=False)

    kept = sorted(set(class_map.values()))
    dropped = sorted(set(our_names) - set(class_map.keys()))
    print(f"Built COCO-overlap eval set at {tmp_dir} — classes: {kept}"
          + (f" ({', '.join(dropped)} excluded, no COCO equivalent)" if dropped else ""))
    return str(out_path)
