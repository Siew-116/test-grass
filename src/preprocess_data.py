# ============================================================
# FULL PIPELINE
# 1. Generate masks from annotations
# 2. Class balance analysis
# 3. Stratified train/val/test split (80/10/10)
# ============================================================

import os
import json
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_FOLDERS = [
    r"C:\Users\ASUS\Desktop\1_DATASET_GRASS\grass_dataset\dataset_01",
    r"C:\Users\ASUS\Desktop\1_DATASET_GRASS\grass_dataset\dataset_02",
    r"C:\Users\ASUS\Desktop\1_DATASET_GRASS\grass_dataset\dataset_03",
    r"C:\Users\ASUS\Desktop\1_DATASET_GRASS\grass_dataset\dataset_04",
    r"C:\Users\ASUS\Desktop\1_DATASET_GRASS\grass_dataset\dataset_05",
    r"C:\Users\ASUS\Desktop\1_DATASET_GRASS\grass_dataset\dataset_06"
]

OUTPUT_ROOT  = r"C:\Users\ASUS\Desktop\1_DATASET_GRASS\grass_dataset"
IMAGE_DIR    = os.path.join(OUTPUT_ROOT, "images")
MASK_DIR     = os.path.join(OUTPUT_ROOT, "masks")
SPLIT_DIR    = os.path.join(OUTPUT_ROOT, "split_3")

PROGRESS_FILE = os.path.join(OUTPUT_ROOT, "analysis_progress.json")
REPORT_FILE   = os.path.join(OUTPUT_ROOT, "class_balance_report.json")
PLOT_FILE     = os.path.join(OUTPUT_ROOT, "class_balance_analysis.png")

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10
RANDOM_SEED  = 42

CLASS_MAP = {
    "grass_short"  : 1,
    "grass_medium" : 2,
    "grass_long"   : 3
}

CLASS_COLOR = {
    "grass_short"  : (0, 255, 0),
    "grass_medium" : (0, 255, 255),
    "grass_long"   : (0, 0, 255)
}

CLASS_NAMES = {
    0: "background",
    1: "grass_short",
    2: "grass_medium",
    3: "grass_long"
}

# ============================================================
# SETUP FOLDERS
# ============================================================
for folder in [IMAGE_DIR, MASK_DIR]:
    os.makedirs(folder, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(SPLIT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(SPLIT_DIR, split, "masks"),  exist_ok=True)
print(f"  Images       : {IMAGE_DIR}")
print(f"  Masks        : {MASK_DIR}")
print(f"  Split output : {SPLIT_DIR}\n")

## -- Parse annotated polygon points to pixel coordinates --
def parse_points(points, width, height):
    # less than 3 cannot form polygons
    if not points or len(points) < 3:
        return None

    # Check normalization (0-1)
    sample_x, sample_y = float(points[0][0]), float(points[0][1])
    is_normalized = (0.0 <= sample_x <= 1.0) and (0.0 <= sample_y <= 1.0)

    parsed = []
    for p in points:
        x, y = float(p[0]), float(p[1])
        if is_normalized:
            px = int(x * width)
            py = int(y * height)
        else:
            px = int(x)
            py = int(y)
        
        # Prevent coord go outside image boundary
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        # Add to parsed coord list
        parsed.append([px, py])

    # Shape: (N rows,2 col) numpy array 
    return np.array(parsed, dtype=np.int32)

# ============================================================
# STAGE 1: GENERATE MASKS
# ============================================================
print("=" * 60)
print("STAGE 1 — GENERATE MASKS")
print("=" * 60)

total_processed = 0
total_skipped   = 0

for dataset_folder in DATASET_FOLDERS:
    # Load annotation files
    print(f"\nProcessing: {dataset_folder}")
    json_path = os.path.join(dataset_folder, "annotations.json")
    if not os.path.exists(json_path):
        print("  [SKIP] No annotations.json found")
        continue

    with open(json_path, "r") as f:
        annotations_data = json.load(f)

    annotations = annotations_data.get("annotations", [])
    print(f"  Found {len(annotations)} annotation entries")

    # Load label names found
    all_labels = set()
    for item in annotations:
        for poly in item.get("polygons", []):
            if poly.get("label"):
                all_labels.add(poly.get("label"))
    print(f"  Labels in JSON     : {all_labels}")
    print(f"  Labels in CLASS_MAP: {set(CLASS_MAP.keys())}")
    unmatched = all_labels - set(CLASS_MAP.keys())
    if unmatched:
        print(f"  [WARNING] Unmatched labels (will be skipped): {unmatched}")

    # Found image data with media name item
    for item in annotations:
        image_name = item.get("mediaName")
        if not image_name:
            continue

        image_path = None
        for root, dirs, files in os.walk(dataset_folder):
            if image_name in files:
                image_path = os.path.join(root, image_name)
                break
    
        if image_path is None:
            print(f"  [MISSING] {image_name}")
            total_skipped += 1
            continue
        
        # Load image with OpenCv
        image = cv2.imread(image_path)
        if image is None:
            print(f"  [ERROR] Cannot read {image_path}")
            total_skipped += 1
            continue
        
        height, width = image.shape[:2] # get image dimensions
        mask = np.zeros((height, width), dtype=np.uint8)    # initialise mask with class 0 (bg)

        for poly in item.get("polygons", []):
            # Get label name for each polygon
            label = poly.get("label")
            if label not in CLASS_MAP:
                continue
            
            # get class id with label
            class_id     = CLASS_MAP[label]

            # Parse points into coordinates
            points       = poly.get("points", [])
            pixel_points = parse_points(points, width, height)
            if pixel_points is None:
                continue
            
            # Fill mask with class ID value on annotated coords
            cv2.fillPoly(mask, [pixel_points], class_id)

        # Save mask data
        shutil.copy2(image_path, os.path.join(IMAGE_DIR, image_name))
        mask_name = os.path.splitext(image_name)[0] + ".png"
        cv2.imwrite(os.path.join(MASK_DIR, mask_name), mask)

        # Get mask value
        unique_vals = np.unique(mask)
        print(f"  Done {image_name} | mask classes: {unique_vals}")
        total_processed += 1

print(f"\nStage 1 complete — Processed: {total_processed}  Skipped: {total_skipped}")

# ============================================================
# STAGE 2: CLASS BALANCE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STAGE 2 — CLASS BALANCE ANALYSIS")
print("=" * 60)

def load_progress():
    # Load progress file
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return None

def save_progress(data):
    # Save progress file as JSON
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Sort mask files alphabetically
mask_files  = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".png")])
total_masks = len(mask_files)
print(f"Found {total_masks} mask files")

progress = load_progress()
if progress:
    # Restore all previously computed data
    already_processed = set(r["file"] for r in progress["per_image_results"])
    pixel_counts      = {int(k): v for k, v in progress["pixel_counts"].items()}
    image_counts      = {int(k): v for k, v in progress["image_counts"].items()}
    per_image_results = progress["per_image_results"]
    failed_files      = progress["failed_files"]
    combination_list  = [frozenset(c) for c in progress["combination_list"]]
    print(f"Resuming from saved progress -- {len(already_processed)} already done\n")
else:
    already_processed = set() # filename done processing
    pixel_counts      = {0: 0, 1: 0, 2: 0, 3: 0} # total pixels per class
    image_counts      = {0: 0, 1: 0, 2: 0, 3: 0} # total img per class
    per_image_results = []
    failed_files      = []
    combination_list  = []
    print("No saved progress found -- starting fresh\n")

remaining = [f for f in mask_files if f not in already_processed]
print(f"Remaining to process: {len(remaining)}\n")

for i, mask_file in enumerate(remaining):
    print(f"  [{i+1:>4}/{len(remaining)}] Processing: {mask_file}", end=" ")
    mask_path = os.path.join(MASK_DIR, mask_file)

    # Load mask as grayscale
    mask      = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"-- FAILED to read")
        failed_files.append(mask_file)
    else:
        # Get unique class ID in mask
        unique = set(np.unique(mask).tolist())
        combination_list.append(frozenset(unique)) # track class combination overall

        per_image_pixel = {}
        for class_id in CLASS_NAMES:
            # Count number of pixels in mask
            count = int(np.sum(mask == class_id))
            # Add to global total pixels in all mask
            pixel_counts[class_id] += count
            per_image_pixel[class_id] = count # store for this mask
            # Image counter for this class
            if class_id in unique:
                image_counts[class_id] += 1

        grass_classes = unique - {0}
        print(f"-- classes: {sorted(unique)}  grass: {sorted(grass_classes)}")

        # Store result per image
        per_image_results.append({
            "file"          : mask_file,
            "shape"         : list(mask.shape),
            "unique_classes": sorted(unique),
            "grass_classes" : sorted(grass_classes),
            "pixel_counts"  : {CLASS_NAMES[k]: per_image_pixel[k] for k in CLASS_NAMES}
        })

    # Save progress after every img
    save_progress({
        "last_updated"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pixel_counts"     : pixel_counts,
        "image_counts"     : image_counts,
        "per_image_results": per_image_results,
        "failed_files"     : failed_files,
        "combination_list" : [sorted(c) for c in combination_list]
    })

print(f"\nDone. {len(per_image_results)} processed, {len(failed_files)} failed\n")

# Level 1 - Image level (how many images each class have)
print("===== LEVEL 1 — IMAGE LEVEL =====")
print(f"{'Class':<5} {'Name':<15} {'Images':>8} {'% of total':>12}  Status")
print("-" * 55)

level1_results = {}
for class_id, name in CLASS_NAMES.items():
    count = image_counts[class_id]
    pct   = count / total_masks * 100
    if pct < 20:
        status = "TOO RARE"
    elif pct > 95 and class_id != 0:
        status = "ABNORMAL"
    else:
        status = "NORMAL"
    print(f"  {class_id:<3} {name:<15} {count:>8} {pct:>11.1f}%  {status}")
    level1_results[name] = {"image_count": count, "percentage": round(pct, 2), "status": status}

# Level 2 - Pixel level (how many pixels each class have)
grand_total = sum(pixel_counts.values()) # total pixel for all masks
print("\n===== LEVEL 2 — PIXEL LEVEL =====")
print(f"{'Class':<5} {'Name':<15} {'Pixels':>12} {'% of total':>12}  Status")
print("-" * 60)

level2_results = {}
for class_id, name in CLASS_NAMES.items():
    count = pixel_counts[class_id]
    pct   = count / grand_total * 100
    if class_id == 0:
        status = "background (expected)"
    elif pct < 5:
        status = "TOO FEW PIXELS"
    elif pct < 15:
        status = "LOW"
    else:
        status = "NORMAL"
    print(f"  {class_id:<3} {name:<15} {count:>12,} {pct:>11.1f}%  {status}")
    level2_results[name] = {"pixel_count": count, "percentage": round(pct, 2), "status": status}

# Level 3 - Combination distribution (how many image have same class combination)
combo_counter = Counter(combination_list)
print("\n===== LEVEL 3 — COMBINATION DISTRIBUTION =====")
print(f"{'Combination':<45} {'Count':>6} {'%':>8}")
print("-" * 62)

level3_results = []
for combo, count in sorted(combo_counter.items(), key=lambda x: -x[1]):
    classes_str = "{" + ", ".join(str(c) for c in sorted(combo)) + "}"
    names_str   = "+".join(CLASS_NAMES[c] for c in sorted(combo) if c != 0)
    if not names_str:
        names_str = "background only"
    label = f"{classes_str} ({names_str})"
    pct   = count / total_masks * 100
    print(f"  {label:<43} {count:>6}  {pct:>6.1f}%")
    level3_results.append({
        "combination": sorted(combo),
        "label"      : label,
        "count"      : count,
        "percentage" : round(pct, 2)
    })

# Get majority and minority class (ignore bg)
grass_pixel_pcts = {k: pixel_counts[k] / grand_total * 100 for k in [1, 2, 3]}
min_class = min(grass_pixel_pcts, key=grass_pixel_pcts.get)
max_class = max(grass_pixel_pcts, key=grass_pixel_pcts.get)
ratio     = grass_pixel_pcts[max_class] / max(grass_pixel_pcts[min_class], 0.001)

print("\n===== Summary (pixel level) =====")
print(f"  Most dominant grass class  : {CLASS_NAMES[max_class]} ({grass_pixel_pcts[max_class]:.1f}%)")
print(f"  Least dominant grass class : {CLASS_NAMES[min_class]} ({grass_pixel_pcts[min_class]:.1f}%)")
print(f"  Imbalance ratio            : {ratio:.1f}x")

# Save report
report = {
    "generated_at"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "mask_directory"    : MASK_DIR,
    "total_masks"       : total_masks,
    "failed_files"      : failed_files,
    "level1_image"      : level1_results,
    "level2_pixel"      : level2_results,
    "level3_combination": level3_results,
    "recommendation"    : {
        "most_dominant_grass"  : CLASS_NAMES[max_class],
        "least_dominant_grass" : CLASS_NAMES[min_class],
        "imbalance_ratio"      : round(ratio, 2),
        "imbalance_level"      : imbalance_level,
        "recommendations"      : recommendations
    },
    "per_image"         : per_image_results
}

with open(REPORT_FILE, "w") as f:
    json.dump(report, f, indent=4)
print(f"\nReport saved : {REPORT_FILE}")

if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
    print(f"Progress file cleared : {PROGRESS_FILE}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Class Balance Analysis -- {total_masks} masks", fontsize=13, fontweight="bold")

colors = ["gray", "green", "yellow", "red"]
names  = [CLASS_NAMES[i] for i in range(4)]

img_pcts = [image_counts[i] / total_masks * 100 for i in range(4)]
bars = axes[0].bar(names, img_pcts, color=colors, edgecolor="black")
axes[0].set_title("Image Level\n(% images containing class)")
axes[0].set_ylabel("% of images")
axes[0].set_ylim(0, 110)
for bar, v in zip(bars, img_pcts):
    axes[0].text(bar.get_x() + bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=9)

pix_counts_plot = [pixel_counts[i] for i in range(4)]
axes[1].pie(pix_counts_plot, labels=names, colors=colors, autopct="%1.1f%%", startangle=140)
axes[1].set_title("Pixel Level\n(% pixels per class)")

combo_labels_plot = []
combo_counts_plot = []
for combo, count in sorted(combo_counter.items(), key=lambda x: -x[1]):
    classes_str = "{" + ",".join(str(c) for c in sorted(combo)) + "}"
    combo_labels_plot.append(classes_str)
    combo_counts_plot.append(count)

axes[2].barh(combo_labels_plot, combo_counts_plot, color="steelblue", edgecolor="black")
axes[2].set_title("Combination Distribution")
axes[2].set_xlabel("Number of images")
for i, v in enumerate(combo_counts_plot):
    axes[2].text(v + 0.1, i, str(v), va="center", fontsize=9)

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()
print(f"Plot saved : {PLOT_FILE}")

# ============================================================
# STAGE 3: STRATIFIED TRAIN/VAL/TEST SPLIT (80/10/10)
# ============================================================
print("\n" + "=" * 60)
print("STAGE 3 - STRATIFIED TRAIN/VAL/TEST SPLIT (80/10/10)")
print("=" * 60)

# Build file list based on class distribution report
all_files    = []
combo_labels = []
for item in report["per_image"]:
    # convert classes list to string key used for stratify label
    combo = str(tuple(sorted(item["unique_classes"])))
    all_files.append(item["file"])
    combo_labels.append(combo)

total        = len(all_files)
combo_counts = Counter(combo_labels)

print(f"\nTotal images : {total}")
print(f"Target split : Train {TRAIN_RATIO*100:.0f}% / Val {VAL_RATIO*100:.0f}% / Test {TEST_RATIO*100:.0f}%")
print(f"\nCombination distribution:")
print(f"  {'Combination':<30} {'Count':>6} {'%':>7}")
print(f"  {'-'*47}")
for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1]):
    print(f"  {combo:<30} {count:>6} {count/total*100:>6.1f}%")

# ==========================================
# EDGE CASE HANDLING (at least 2 samples per group is needed for stratification)
# ==========================================
# Groups with 1 image  -> force to train only
# Groups with 2 images -> 1 train, 1 test
# Groups with 3 images -> 2 train, 1 test
# Groups with 4+ images -> stratified split normally

# Identify group
one_img_groups   = {c for c, n in combo_counts.items() if n == 1}
two_img_groups   = {c for c, n in combo_counts.items() if n == 2}
three_img_groups = {c for c, n in combo_counts.items() if n == 3}

force_train_files = []
force_val_files   = []
force_test_files  = []
splittable_files  = [] # can use normal stratified split
splittable_labels = []

two_img_tracker   = defaultdict(list)
three_img_tracker = defaultdict(list)

for f, label in zip(all_files, combo_labels):
    if label in one_img_groups:
        force_train_files.append(f) # only 1 -> forced to train set
    elif label in two_img_groups:
        two_img_tracker[label].append(f) # only 2 -> manual 1,1 split
    elif label in three_img_groups:
        three_img_tracker[label].append(f) # only 3 -> manual 2,1 split
    else:
        # Normal
        splittable_files.append(f)
        splittable_labels.append(label)

# 2-image groups: first to train, second to test
for label, files in two_img_tracker.items():
    force_train_files.append(files[0])
    force_test_files.append(files[1])

# 3-image groups: first 2 to train, last to test
for label, files in three_img_tracker.items():
    force_train_files.append(files[0])
    force_train_files.append(files[1])
    force_test_files.append(files[2])

print(f"  Forced to train                               : {len(force_train_files)}")
print(f"  Forced to val                                 : {len(force_val_files)}")
print(f"  Forced to test                                : {len(force_test_files)}")
print(f"  Going through stratified split                : {len(splittable_files)}")

assert len(force_train_files) + len(force_val_files) + len(force_test_files) + len(splittable_files) == total, \
    "ERROR: counts dont add up!"
print(f"\n  Sanity check passed: {total} total accounted for")

# ==========================================
# STRATIFIED SPLITTING
# Step 1: split off test (10%)
# Step 2: split remaining into train (80%) and val (10%)
# ==========================================

if len(splittable_files) > 0:

    # Step 1 — separate test (10%)
    trainval_files, test_files, trainval_labels, _ = train_test_split(
        splittable_files,
        splittable_labels,
        test_size    = TEST_RATIO,
        random_state = RANDOM_SEED,
        stratify     = splittable_labels # Stratify based on class label
    )

    # Step 2 — split trainval into train and val
    # val adjusted ratio = 10 / (80 + 10) = 0.111
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    train_files, val_files, _, _ = train_test_split(
        trainval_files,
        trainval_labels,
        test_size    = val_ratio_adjusted,
        random_state = RANDOM_SEED,
        stratify     = trainval_labels
    )

else:
    train_files = []
    val_files   = []
    test_files  = []

# Combine with forced files
train_files = list(train_files) + force_train_files
val_files   = list(val_files)   + force_val_files
test_files  = list(test_files)  + force_test_files

print(f"\nFinal split:")
print(f"  Train : {len(train_files)} ({len(train_files)/total*100:.1f}%)")
print(f"  Val   : {len(val_files)}   ({len(val_files)/total*100:.1f}%)")
print(f"  Test  : {len(test_files)}  ({len(test_files)/total*100:.1f}%)")
print(f"  Total : {len(train_files)+len(val_files)+len(test_files)}")

# Per combination verification
print(f"\nPer combination split:")
print(f"  {'Combination':<30} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
print(f"  {'-'*58}")

file_to_combo      = dict(zip(all_files, combo_labels))
combo_train_counts = Counter(file_to_combo[f] for f in train_files)
combo_val_counts   = Counter(file_to_combo[f] for f in val_files)
combo_test_counts  = Counter(file_to_combo[f] for f in test_files)

for combo in sorted(combo_counts.keys()):
    t  = combo_counts[combo]
    tr = combo_train_counts.get(combo, 0)
    va = combo_val_counts.get(combo, 0)
    te = combo_test_counts.get(combo, 0)
    print(f"  {combo:<30} {t:>6} {tr:>6} {va:>6} {te:>6}")

# ==========================================
# COPY FILES TO SPLIT FOLDERS
# ==========================================
def copy_pair(mask_file, split_name, index, total_count):
    image_file = os.path.splitext(mask_file)[0] + ".jpg"
    src_mask   = os.path.join(MASK_DIR,  mask_file)
    src_image  = os.path.join(IMAGE_DIR, image_file)
    dst_mask   = os.path.join(SPLIT_DIR, split_name, "masks",  mask_file)
    dst_image  = os.path.join(SPLIT_DIR, split_name, "images", image_file)

    shutil.copy2(src_mask, dst_mask)

    if os.path.exists(src_image):
        shutil.copy2(src_image, dst_image)
        status = "OK"
    else:
        status = f"MISSING IMAGE: {image_file}"

    print(f"  [{index:>4}/{total_count}] {mask_file} -- {status}")
    return status

print("\nCopying train files...")
train_errors = []
for i, f in enumerate(train_files):
    result = copy_pair(f, "train", i+1, len(train_files))
    if result != "OK":
        train_errors.append({"file": f, "error": result})

print("\nCopying val files...")
val_errors = []
for i, f in enumerate(val_files):
    result = copy_pair(f, "val", i+1, len(val_files))
    if result != "OK":
        val_errors.append({"file": f, "error": result})

print("\nCopying test files...")
test_errors = []
for i, f in enumerate(test_files):
    result = copy_pair(f, "test", i+1, len(test_files))
    if result != "OK":
        test_errors.append({"file": f, "error": result})

# ==========================================
# VERIFY DISTRIBUTION
# ==========================================

def get_distribution(split_name):
    mask_dir   = os.path.join(SPLIT_DIR, split_name, "masks")
    pix_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    img_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    files      = os.listdir(mask_dir)
    n_masks    = 0

    for j, mask_file in enumerate(files):
        print(f"  Verifying {split_name} [{j+1:>4}/{len(files)}]", end="\r")
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        unique = set(np.unique(mask).tolist())
        for class_id in pix_counts:
            pix_counts[class_id] += int(np.sum(mask == class_id))
            if class_id in unique:
                img_counts[class_id] += 1
        n_masks += 1

    print()
    return pix_counts, img_counts, n_masks

print("\n===== VERIFICATION =====")

split_report = {}
for split_name in ["train", "val", "test"]:
    pix_counts, img_counts, n_masks = get_distribution(split_name)
    grand = sum(pix_counts.values())

    print(f"\n  {split_name.upper()} ({n_masks} images)")
    print(f"  {'Class':<5} {'Name':<15} {'Images':>8} {'Img%':>7} {'Pixel%':>8}")
    print(f"  {'-'*50}")

    split_report[split_name] = {"n_images": n_masks, "classes": {}}
    for class_id, name in CLASS_NAMES.items():
        img_pct = img_counts[class_id] / n_masks * 100 if n_masks > 0 else 0
        pix_pct = pix_counts[class_id] / grand   * 100 if grand   > 0 else 0
        print(f"  {class_id:<5} {name:<15} {img_counts[class_id]:>8} "
              f"{img_pct:>6.1f}%  {pix_pct:>6.1f}%")
        split_report[split_name]["classes"][name] = {
            "image_count"      : img_counts[class_id],
            "image_percentage" : round(img_pct, 2),
            "pixel_percentage" : round(pix_pct, 2)
        }

# ==========================================
# PLOT SPLIT DISTRIBUTION
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Split Distribution Verification (80/10/10)", fontsize=13, fontweight="bold")

x     = np.arange(len(CLASS_NAMES))
width = 0.25

for ax, metric, title in zip(
    axes,
    ["image_percentage", "pixel_percentage"],
    ["Image Level (% images containing class)",
     "Pixel Level (% pixels per class)"]
):
    train_vals = [split_report["train"]["classes"][CLASS_NAMES[i]][metric] for i in range(4)]
    val_vals   = [split_report["val"]["classes"][CLASS_NAMES[i]][metric]   for i in range(4)]
    test_vals  = [split_report["test"]["classes"][CLASS_NAMES[i]][metric]  for i in range(4)]

    ax.bar(x - width,  train_vals, width, label="Train", color="steelblue",  edgecolor="black")
    ax.bar(x,          val_vals,   width, label="Val",   color="seagreen",   edgecolor="black")
    ax.bar(x + width,  test_vals,  width, label="Test",  color="darkorange", edgecolor="black")

    ax.set_title(title)
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_NAMES[i] for i in range(4)], rotation=15)
    ax.legend()

plt.tight_layout()
split_plot_path = os.path.join(OUTPUT_ROOT, "split_distribution.png")
plt.savefig(split_plot_path, dpi=150)
plt.show()
print(f"\nPlot saved : {split_plot_path}")

# ==========================================
# SAVE SPLIT REPORT
# ==========================================

report_out = {
    "generated_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "method"       : "stratified train/val/test split 80/10/10",
    "config"       : {
        "train_ratio"        : TRAIN_RATIO,
        "val_ratio"          : VAL_RATIO,
        "test_ratio"         : TEST_RATIO,
        "random_seed"        : RANDOM_SEED,
        "total_images"       : total,
        "forced_train_count" : len(force_train_files),
        "forced_val_count"   : len(force_val_files),
        "forced_test_count"  : len(force_test_files)
    },
    "train_files"  : sorted(train_files),
    "val_files"    : sorted(val_files),
    "test_files"   : sorted(test_files),
    "train_errors" : train_errors,
    "val_errors"   : val_errors,
    "test_errors"  : test_errors,
    "distribution" : split_report
}

split_report_path = os.path.join(OUTPUT_ROOT, "split_report.json")
with open(split_report_path, "w") as f:
    json.dump(report_out, f, indent=4)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"\nStage 1 — Mask generation")
print(f"  Processed : {total_processed}")
print(f"  Skipped   : {total_skipped}")
print(f"\nStage 2 — Class balance analysis")
print(f"  Total masks    : {total_masks}")
print(f"  Imbalance level: {imbalance_level}")
print(f"  Report saved   : {REPORT_FILE}")
print(f"\nStage 3 — Train/Val/Test split (80/10/10)")
print(f"  Method  : Stratified by class combination (seed={RANDOM_SEED})")
print(f"  Train   : {len(train_files)} ({len(train_files)/total*100:.1f}%)")
print(f"  Val     : {len(val_files)}   ({len(val_files)/total*100:.1f}%)")
print(f"  Test    : {len(test_files)}  ({len(test_files)/total*100:.1f}%)")
print(f"  Report  : {split_report_path}")
print(f"\nOutput folders:")
print(f"  Train images : {os.path.join(SPLIT_DIR, 'train', 'images')}")
print(f"  Train masks  : {os.path.join(SPLIT_DIR, 'train', 'masks')}")
print(f"  Val images   : {os.path.join(SPLIT_DIR, 'val',   'images')}")
print(f"  Val masks    : {os.path.join(SPLIT_DIR, 'val',   'masks')}")
print(f"  Test images  : {os.path.join(SPLIT_DIR, 'test',  'images')}")
print(f"  Test masks   : {os.path.join(SPLIT_DIR, 'test',  'masks')}")
