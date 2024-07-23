import cv2
import numpy as np
from ultralytics import YOLO

def process_video(video_path):
    # Load YOLO model
    model = YOLO('yolov8n.pt')

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frames_processed = 0
    detections = []
    player_tracks = {}
    next_player_id = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        frame_detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                if cls == 0:  # Class 0 is person in COCO dataset
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Simple tracking: assign ID based on proximity to previous detections
                    closest_id = None
                    min_distance = float('inf')
                    for player_id, track in player_tracks.items():
                        if len(track) > 0:
                            prev_x, prev_y = track[-1]
                            distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_id = player_id

                    if closest_id is None or min_distance > 50:  # Threshold for new player
                        closest_id = next_player_id
                        next_player_id += 1
                        player_tracks[closest_id] = []

                    player_tracks[closest_id].append((center_x, center_y))
                    
                    frame_detections.append({
                        'type': 'player',
                        'id': closest_id,
                        'box': box.tolist(),
                        'confidence': float(conf)
                    })
                elif cls == 32:  # Class 32 is sports ball in COCO dataset
                    frame_detections.append({
                        'type': 'ball',
                        'box': box.tolist(),
                        'confidence': float(conf)
                    })

        detections.append({
            'frame': frames_processed,
            'detections': frame_detections
        })

        frames_processed += 1
        if frames_processed % 100 == 0:
            print(f"Processed {frames_processed} frames")

    cap.release()

    # Basic event detection
    events = detect_events(detections)

    return {'detections': detections, 'events': events}

def detect_events(detections):
    events = []
    for i in range(1, len(detections)):
        prev_frame = detections[i-1]
        curr_frame = detections[i]
        
        # Detect passes
        ball_prev = next((d for d in prev_frame['detections'] if d['type'] == 'ball'), None)
        ball_curr = next((d for d in curr_frame['detections'] if d['type'] == 'ball'), None)
        
        if ball_prev and ball_curr:
            ball_movement = np.linalg.norm(np.array(ball_curr['box'][:2]) - np.array(ball_prev['box'][:2]))
            if ball_movement > 50:  # Threshold for significant ball movement
                events.append({
                    'type': 'possible_pass',
                    'frame': curr_frame['frame']
                })

    return events