import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rfdetr import RFDETRMedium

def convert_to_coco_format(image_id, boxes_xyxy, scores, labels, category_mapping):
    """Converts xyxy boxes to COCO format: [x, y, width, height]"""
    results = []
    for box, score, label in zip(boxes_xyxy, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        
        # Map model class ID back to COCO category ID
        coco_cat_id = category_mapping[int(label)]
        
        results.append({
            "image_id": image_id,
            "category_id": coco_cat_id,
            "bbox": [x1, y1, w, h],
            "score": float(score)
        })
    return results

def evaluate_predictions(coco_gt, predictions_file):
    """Runs COCO mAP evaluation on a saved predictions JSON."""
    coco_dt = coco_gt.loadRes(predictions_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0] # Return mAP @ IoU=0.50:0.95

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--npy_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="Path to your trained .pth weights")
    parser.add_argument("--resolution", type=int, default=640)
    args = parser.parse_args()

    # 1. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rfdetr = RFDETRMedium(pretrain_weights=args.weights, resolution=args.resolution)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = rfdetr.model.model.to(device)
    model.eval()

    # 2. Load COCO Ground Truth and Metadata
    coco_gt = COCO(args.annotation_file)
    
    # Map model 0-indexed classes to COCO arbitrary category IDs
    cat_ids = sorted(coco_gt.getCatIds())
    category_mapping = {i: cat_id for i, cat_id in enumerate(cat_ids)}
    
    metadata_path = os.path.join(args.npy_dir, "mlperf_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    results_raw = []
    results_npy = []
    
    print("Running dual inference...")
    
    for img_id in tqdm(coco_gt.getImgIds(), desc="Evaluating"):
        img_info = coco_gt.loadImgs(img_id)[0]
        meta = metadata[str(img_id)]
        img_path = meta["file_name"]
        
        # ---------------------------------------------------------
        # PATH A: Native .predict() with Raw Image
        # ---------------------------------------------------------
        # Native predict handles everything and returns supervision.Detections
        preds = rfdetr.predict(img_path, threshold=0.001) # Low threshold for mAP calculation
        
        # ---> THE FIX for PATH A: Filter out background/invalid class IDs <---
        num_valid_classes = len(category_mapping)
        
        # Create a boolean mask of valid class IDs
        valid_mask = preds.class_id < num_valid_classes
        
        # Apply the mask directly to the Detections object (supervision supports this)
        preds = preds[valid_mask]

        results_raw.extend(
            convert_to_coco_format(img_id, preds.xyxy, preds.confidence, preds.class_id, category_mapping)
        )
        
        # ---------------------------------------------------------
        # PATH B: Manual Forward Pass with Preprocessed .npy
        # ---------------------------------------------------------
        npy_path = os.path.join(args.npy_dir, f"{img_id}.npy")
        tensor = torch.from_numpy(np.load(npy_path)).unsqueeze(0).to(device) # Shape: [1, 3, 640, 640]
        
        with torch.no_grad():
            outputs = model(tensor) # Raw model forward pass
            
        # Extract DETR heads
        logits = outputs['pred_logits'][0] # (num_queries, num_classes)
        boxes = outputs['pred_boxes'][0]   # (num_queries, 4) in cxcywh [0, 1] format
        
        # Calculate scores and predicted labels
        probs = logits.sigmoid()
        scores, labels = probs.max(dim=-1)
        
        # 1. Filter out background class
        num_valid_classes = len(category_mapping)
        valid_class_mask = labels < num_valid_classes
        boxes = boxes[valid_class_mask]
        scores = scores[valid_class_mask]
        labels = labels[valid_class_mask]
        
        # 2. Filter by score threshold (0.001 for mAP calculation)
        keep = scores > 0.001
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # ---------------------------------------------------------
        # REPLACED LETTERBOX UN-PADDING WITH DIRECT UNSCALING
        # ---------------------------------------------------------
        # Get original image dimensions from the metadata dictionary
        orig_h, orig_w = meta["orig_shape"]
        
        # Unbind normalized cxcywh coordinates (range 0 to 1)
        cx, cy, w, h = boxes.unbind(-1)
        
        # Convert normalized cxcywh directly to absolute xyxy on the original image dimensions
        x1 = (cx - 0.5 * w) * orig_w
        y1 = (cy - 0.5 * h) * orig_h
        x2 = (cx + 0.5 * w) * orig_w
        y2 = (cy + 0.5 * h) * orig_h
        
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        # ---------------------------------------------------------
        
        results_npy.extend(
            convert_to_coco_format(
                img_id, 
                boxes_xyxy.cpu().numpy(), 
                scores.cpu().numpy(), 
                labels.cpu().numpy(), 
                category_mapping
            )
        )

    # 3. Save and Evaluate
    print("\nSaving predictions...")
    with open("preds_raw.json", "w") as f: json.dump(results_raw, f)
    with open("preds_npy.json", "w") as f: json.dump(results_npy, f)
        
    print("\n=== Evaluating Native .predict() (Raw Images) ===")
    map_raw = evaluate_predictions(coco_gt, "preds_raw.json")
    
    print("\n=== Evaluating Manual Forward Pass (.npy tensors) ===")
    map_npy = evaluate_predictions(coco_gt, "preds_npy.json")
    
    print("\n=== SUMMARY ===")
    print(f"Native predict() mAP:  {map_raw:.8f}")
    print(f"Offline .npy mAP:      {map_npy:.8f}")
    
    if abs(map_raw - map_npy) < 0.001:
        print("✅ SUCCESS: Both pathways yielded identical accuracy. Your MLPerf preprocessing is correct.")
    else:
        print("❌ WARNING: Accuracies differ. Check normalization or scaling logic.")

if __name__ == "__main__":
    main()