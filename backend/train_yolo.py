from ultralytics import YOLO
from roboflow import Roboflow
import os

def train_yolo_model():
    # Initialize Roboflow
    rf = Roboflow(api_key="1cwGoU29yCb6DDrXfvwL")
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    dataset = project.version(1).download("yolov8")

    # Print dataset location for verification
    print(f"Dataset downloaded to: {dataset.location}")

    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data=dataset.location + '/data.yaml',  # Use the downloaded dataset location
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

if __name__ == "__main__":
    train_yolo_model()
    print("Model training complete. You can now use the trained model in video_processor.py")