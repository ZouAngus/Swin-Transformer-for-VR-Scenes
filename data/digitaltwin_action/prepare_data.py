"""
Prepare digitaltwin_action data for MMAction2 training.
- Parse labels from filenames
- Split train/val (80/20, stratified by class)
- Generate annotation txt files
"""
import os
import glob
import random

random.seed(42)

import os as _os
DATA_DIR = _os.path.dirname(_os.path.abspath(__file__))
OUT_DIR = DATA_DIR

# Label mapping: action keyword in filename -> (class_index, class_name)
LABEL_MAP = {
    "stand": (0, "Stand"),
    "raising_hand": (1, "Raising_hand"),
    "jumping": (2, "Jumping"),
    "squatting": (3, "Squatting"),
}

def extract_label(filename):
    """Extract action label from filename like 01-boss-diagonal-jumping_center-rep1.mp4"""
    fname = os.path.splitext(filename)[0]
    # Split by '-' and find the action part (4th segment: e.g. 'jumping_center')
    parts = fname.split("-")
    # The action keyword is at the start of the 4th part (index 3)
    if len(parts) >= 4:
        action_part = parts[3]  # e.g. "jumping_center" or "raising_hand_center" or "stand_center"
        # Check each label keyword
        for keyword, (idx, name) in LABEL_MAP.items():
            if action_part.startswith(keyword):
                return idx, name
    return None, None

# Collect all videos with labels
samples = []  # (relative_path, class_index, class_name)
for root, dirs, files in os.walk(DATA_DIR):
    for f in sorted(files):
        if not f.endswith(".mp4"):
            continue
        label_idx, label_name = extract_label(f)
        if label_idx is None:
            print(f"WARNING: Cannot parse label from {f}")
            continue
        # Relative path from data root (for annotation file)
        rel_dir = os.path.relpath(root, os.path.dirname(DATA_DIR))
        rel_path = os.path.join(rel_dir, f)
        samples.append((rel_path, label_idx, label_name))

print(f"Total samples: {len(samples)}")

# Count per class
from collections import Counter
class_counts = Counter(s[1] for s in samples)
for idx in sorted(class_counts.keys()):
    name = [s[2] for s in samples if s[1] == idx][0]
    print(f"  Class {idx} ({name}): {class_counts[idx]}")

# Stratified split: 80% train, 20% val
train_samples = []
val_samples = []
for cls_idx in sorted(class_counts.keys()):
    cls_samples = [s for s in samples if s[1] == cls_idx]
    random.shuffle(cls_samples)
    split_point = max(1, int(len(cls_samples) * 0.8))
    train_samples.extend(cls_samples[:split_point])
    val_samples.extend(cls_samples[split_point:])

random.shuffle(train_samples)
random.shuffle(val_samples)

print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")

# Write annotation files
train_file = os.path.join(OUT_DIR, "train_list.txt")
val_file = os.path.join(OUT_DIR, "val_list.txt")

with open(train_file, "w") as f:
    for rel_path, cls_idx, _ in train_samples:
        f.write(f"{rel_path} {cls_idx}\n")

with open(val_file, "w") as f:
    for rel_path, cls_idx, _ in val_samples:
        f.write(f"{rel_path} {cls_idx}\n")

# Write class index file
classind_file = os.path.join(OUT_DIR, "classInd.txt")
with open(classind_file, "w") as f:
    for keyword, (idx, name) in sorted(LABEL_MAP.items(), key=lambda x: x[1][0]):
        f.write(f"{idx} {name}\n")

# Write label file for demo
label_file = os.path.join(OUT_DIR, "label_4cls.txt")
with open(label_file, "w") as f:
    for keyword, (idx, name) in sorted(LABEL_MAP.items(), key=lambda x: x[1][0]):
        f.write(f"{name}\n")

print(f"\nFiles written:")
print(f"  {train_file}")
print(f"  {val_file}")
print(f"  {classind_file}")
print(f"  {label_file}")

# Print train/val split per class
train_counts = Counter(s[1] for s in train_samples)
val_counts = Counter(s[1] for s in val_samples)
print(f"\nPer-class split:")
for idx in sorted(class_counts.keys()):
    name = [s[2] for s in samples if s[1] == idx][0]
    print(f"  {name}: train={train_counts[idx]}, val={val_counts[idx]}")
