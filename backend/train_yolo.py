from ultralytics import YOLO
from roboflow import Roboflow
import os

# Use the correct API key from Roboflow
rf = Roboflow(api_key="1cwGoU29yCb6DDrXfvwL")
project = rf.workspace("FootballVideoTrackingApp").project("football-video-tracking-project")
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

# Print the path of the best model
print(f"Best model saved at: {model.best}")
