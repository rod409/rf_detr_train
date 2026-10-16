import os
import json
import argparse
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def convert_to_coco_format(image_id, boxes_xyxy, scores, labels, category_mapping):
    """Converts xyxy boxes to COCO format: [x, y, width, height]"""
    results = []
    for box, score, label in zip(boxes_xyxy, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        
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
    return coco_eval.stats[0]  # Return mAP @ IoU=0.50:0.95

def main():
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR ONNX model on preprocessed .npy files")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to COCO validation JSON")
    parser.add_argument("--npy_dir", type=str, required=True, help="Directory containing preprocessed .npy files")
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to exported .onnx model")
    args = parser.parse_args()

    # 1. Initialize ONNX Runtime Session (CUDA if available, else CPU)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    print(f"Loading ONNX model with providers: {providers}")
    session = ort.InferenceSession(args.onnx_model, providers=providers)

    # 2. Load COCO Ground Truth and Metadata
    coco_gt = COCO(args.annotation_file)
    
    cat_ids = sorted(coco_gt.getCatIds())
    category_mapping = {i: cat_id for i, cat_id in enumerate(cat_ids)}
    num_valid_classes = len(category_mapping)
    
    metadata_path = os.path.join(args.npy_dir, "mlperf_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    results_onnx = []

    print("Running ONNX inference...")
    for img_id in tqdm(coco_gt.getImgIds(), desc="Evaluating ONNX"):
        img_info = coco_gt.loadImgs(img_id)[0]
        meta = metadata[str(img_id)]
        
        npy_path = os.path.join(args.npy_dir, f"{img_id}.npy")
        if not os.path.exists(npy_path):
            continue

        # Load tensor [C, H, W] -> [1, C, H, W]
        tensor = np.load(npy_path).astype(np.float32)
        if tensor.ndim == 3:
            tensor = np.expand_dims(tensor, axis=0)
            
        tensor = np.ascontiguousarray(tensor)

        # Explicitly fetch "dets" and "labels" from input "input"
        outputs = session.run(["dets", "labels"], {"input": tensor})
        boxes_raw = outputs[0][0]   # Shape: (num_queries, 4) in cxcywh normalized format
        logits_raw = outputs[1][0]  # Shape: (num_queries, num_classes) pre-sigmoid

        # Compute probabilities and confidence scores
        probs = sigmoid(logits_raw)
        scores = np.max(probs, axis=-1)
        labels = np.argmax(probs, axis=-1)

        # 1. Filter out background class
        valid_class_mask = labels < num_valid_classes
        boxes = boxes_raw[valid_class_mask]
        scores = scores[valid_class_mask]
        labels = labels[valid_class_mask]

        # 2. Filter by threshold (0.001 for mAP calculation)
        keep = scores > 0.001
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # 3. Direct unscaling using original image dimensions
        orig_h, orig_w = meta["orig_shape"]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        x1 = (cx - 0.5 * w) * orig_w
        y1 = (cy - 0.5 * h) * orig_h
        x2 = (cx + 0.5 * w) * orig_w
        y2 = (cy + 0.5 * h) * orig_h

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)

        results_onnx.extend(
            convert_to_coco_format(img_id, boxes_xyxy, scores, labels, category_mapping)
        )

    # 3. Save and Evaluate
    pred_file = "preds_onnx.json"
    print(f"\nSaving ONNX predictions to {pred_file}...")
    with open(pred_file, "w") as f:
        json.dump(results_onnx, f)

    print("\n=== Evaluating ONNX Model ===")
    map_onnx = evaluate_predictions(coco_gt, pred_file)

    print(f"\nONNX Inference mAP: {map_onnx:.4f}")

if __name__ == "__main__":
    main()