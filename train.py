import os
import argparse

# Disable NVLS to prevent NCCL Fabric Manager errors on this server
# MUST be set before importing rfdetr or torch
os.environ["NCCL_NVLS_ENABLE"] = "0"

from rfdetr import RFDETRMedium

# Set up the command line argument parser
parser = argparse.ArgumentParser(description="Train RF-DETR on a custom dataset")
parser.add_argument(
    "--dataset_dir", 
    type=str, 
    required=True, 
    help="Path to the COCO-formatted dataset directory (e.g., zod_rfdetr)"
)
parser.add_argument(
    "--resolution", 
    type=int, 
    default=640, 
    help="Input resolution as a single integer. Must be divisible by 32. (default: 640)"
)
args = parser.parse_args()

# Enforce the divisibility rule to prevent obscure errors during the forward pass
if args.resolution % 32 != 0:
    raise ValueError(f"Resolution must be divisible by 32. You provided: {args.resolution}")

# Initialize model with the single integer
model = RFDETRMedium(resolution=args.resolution)

model.train(
    dataset_dir=args.dataset_dir,
    epochs=60,
    early_stopping=True,            # Enable early stopping
    early_stopping_patience=10,     # Set the patience threshold here
    batch_size=8,
    grad_accum_steps=1,
    lr=1e-4,
    output_dir="zod_training_output/",
    progress_bar="tqdm",
    devices="auto"
)