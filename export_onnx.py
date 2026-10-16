import argparse
from rfdetr import RFDETRMedium

# Set up the command line argument parser
parser = argparse.ArgumentParser(description="Export an RF-DETR checkpoint to ONNX")
parser.add_argument(
    "checkpoint_path", 
    type=str, 
    help="Path to the trained .pth checkpoint file"
)
parser.add_argument(
    "--output_dir", 
    type=str, 
    default="zod_onnx_export/", 
    help="Directory to save the exported ONNX model"
)

args = parser.parse_args()

# Load the model using the path provided in the terminal
model = RFDETRMedium.from_checkpoint(args.checkpoint_path)

# Export the model
model.export(
    format="onnx",
    shape=(1280,1280),
    output_dir=args.output_dir
)