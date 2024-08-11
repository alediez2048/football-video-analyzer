from ultralytics import YOLO
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov8")

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data=f'{dataset.location}/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='football_players_model'
)

# Validate the model
results = model.val()

# Export the model
model.export(format="onnx")