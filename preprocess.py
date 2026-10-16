import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Preprocess COCO validation dataset for MLPerf RF-DETR")
    parser.add_argument(
        "--annotation_file", 
        type=str, 
        required=True, 
        help="Path to COCO validation JSON containing absolute image paths"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Where to save .npy files"
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=640, 
        help="Target model resolution"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load COCO ground truth JSON
    with open(args.annotation_file, 'r') as f:
        coco_gt = json.load(f)

    images = coco_gt.get("images", [])
    print(f"Loaded {len(images)} images from {args.annotation_file}")

    # ImageNet normalization parameters
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    metadata = {}
    success_count = 0

    for img_entry in tqdm(images, desc="Preprocessing Images", unit="img"):
        img_id = img_entry["id"]
        img_path = img_entry["file_name"]

        if not os.path.exists(img_path):
            tqdm.write(f"Warning: Could not find image at {img_path}, skipping.")
            continue

        try:
            # 1. Open with PIL to match RF-DETR native decoder
            img = Image.open(img_path).convert("RGB")
        except Exception:
            tqdm.write(f"Warning: Failed to read image at {img_path}, skipping.")
            continue

        orig_w, orig_h = img.size  # PIL size is (width, height)

        # 2. Resize using PIL Bilinear matching PyTorch/PIL backend
        img_resized = img.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)

        # 3. Convert to float numpy array and scale to [0, 1]
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        # 4. ImageNet Normalization
        img_tensor = (img_np - mean) / std

        # 5. Transpose HWC -> CHW
        img_tensor = np.transpose(img_tensor, (2, 0, 1))

        # Save binary array using the COCO image ID as the filename
        out_name = f"{img_id}.npy"
        out_path = os.path.join(args.output_dir, out_name)
        np.save(out_path, img_tensor)

        # Record original image dimensions
        metadata[str(img_id)] = {
            "file_name": img_path,
            "orig_shape": [orig_h, orig_w],
            "coco_image_id": img_id
        }
        success_count += 1

    # Save metadata mapping
    with open(os.path.join(args.output_dir, "mlperf_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSuccessfully preprocessed {success_count}/{len(images)} images -> Saved to {args.output_dir}")

if __name__ == "__main__":
    main()