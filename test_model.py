from ultralytics import YOLO
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to best.pt
model_path = os.path.join(current_dir, 'best.pt')

print(f"Attempting to load model from: {model_path}")

try:
    # Load the model
    model = YOLO(model_path)
    
    # Print model information
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model class names: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    
    # Try to run inference on a sample image
    sample_image_path = 'path/to/sample/image.jpg'  # Replace with an actual image path
    if os.path.exists(sample_image_path):
        results = model(sample_image_path)
        print(f"Inference successful. Detected {len(results[0].boxes)} objects.")
    else:
        print(f"Sample image not found at {sample_image_path}")

except Exception as e:
    print(f"Error loading model: {str(e)}")