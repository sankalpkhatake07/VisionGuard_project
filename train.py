import os
import random
import shutil
import zipfile
from ultralytics import YOLO

# ==== STEP 1: Extract ZIP ====
zip_path = "E:\CEP_Project\department_dataset.zip"   # 🔹 Change this
extract_path = "dataset"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✅ Extracted dataset to {extract_path}")
else:
    print(f"✅ Folder '{extract_path}' already exists, using existing data")

# ==== STEP 2: AUTO TRAIN/VAL SPLIT ====
def auto_split_dataset(base_path, split_ratio=0.8):
    img_dir = os.path.join(base_path, "images")
    label_dir = os.path.join(base_path, "labels")

    # Check if split already done
    if os.path.exists(os.path.join(img_dir, "train")):
        print("✅ Train/Val folders already exist, skipping split.")
        return

    print("📂 Splitting dataset into train/val folders...")

    os.makedirs(os.path.join(img_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(label_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(label_dir, "val"), exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_imgs = images[:split_index]
    val_imgs = images[split_index:]

    for img_set, set_name in [(train_imgs, "train"), (val_imgs, "val")]:
        for img_name in img_set:
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(img_dir, set_name, img_name)
            shutil.move(src_img, dst_img)

            # Move corresponding label
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label = os.path.join(label_dir, label_name)
            dst_label = os.path.join(label_dir, set_name, label_name)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)

    print("✅ Dataset split completed.")


auto_split_dataset(extract_path)

# ==== STEP 3: CREATE data.yaml ====
class_file = os.path.join(extract_path, "classes.txt")
with open(class_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

yaml_content = f"""
train: {os.path.abspath(extract_path)}/images/train
val: {os.path.abspath(extract_path)}/images/val

nc: {len(classes)}
names: {classes}
"""

yaml_path = os.path.join(extract_path, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("✅ data.yaml created successfully")

# ==== STEP 4: TRAIN MODEL ====
model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt for better accuracy

model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    name="custom_yolo_model"
)
