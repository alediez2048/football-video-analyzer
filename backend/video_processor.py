import cv2
import numpy as np
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(file_path):
    # Load YOLO model
    model = YOLO('yolov8n.pt')

    # Open the video file
    cap = cv2.VideoCapture(file_path)
    
    frames_processed = 0
    detections = []

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
                frame_detections.append({
                    'type': model.names[int(cls)],
                    'confidence': float(conf),
                    'box': box.tolist()
                })

        detections.append({
            'frame': frames_processed,
            'detections': frame_detections
        })

        frames_processed += 1
        if frames_processed % 100 == 0:
            logger.info(f"Processed {frames_processed} frames")

    cap.release()

    logger.info(f"Total frames processed: {frames_processed}")
    logger.info(f"Total detections: {sum(len(frame['detections']) for frame in detections)}")

    # Basic event detection
    events = detect_events(detections)

    return {
    'total_frames': frames_processed,
    'detections': detections,
    'events': detect_events(detections),
    'field_dimensions': {'width': 105, 'height': 68}  # in meters
}

def detect_events(detections):
    events = []
    last_event_frame = -1
    cooldown_frames = 10
    ball_possession = None
    
    field_width = 105  # standard football field width in meters
    field_height = 68  # standard football field height in meters
    
    def calculate_speed(pos1, pos2, frames_difference):
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        return distance / frames_difference * 24  # Assuming 24 fps, adjust if different
    
    for i in range(1, len(detections)):
        prev_frame = detections[i-1]
        curr_frame = detections[i]
        
        ball_prev = next((d for d in prev_frame['detections'] if d['type'] == 'sports ball'), None)
        ball_curr = next((d for d in curr_frame['detections'] if d['type'] == 'sports ball'), None)
        
        if ball_prev and ball_curr:
            ball_movement = np.linalg.norm(np.array(ball_curr['box'][:2]) - np.array(ball_prev['box'][:2]))
            ball_speed = calculate_speed(ball_curr['box'][:2], ball_prev['box'][:2], 1)
            
            # Detect high-speed ball movement (possible shot or long pass)
            if ball_speed > 20:  # Speed threshold in m/s
                event_type = 'possible_shot' if ball_speed > 30 else 'possible_long_pass'
                events.append({'type': event_type, 'frame': curr_frame['frame'], 'speed': ball_speed})
                logger.info(f"{event_type.capitalize()} detected at frame {curr_frame['frame']} with speed {ball_speed:.2f} m/s")
            
            # Detect change in ball possession
            closest_player = min((d for d in curr_frame['detections'] if d['type'] == 'person'), 
                                 key=lambda x: np.linalg.norm(np.array(x['box'][:2]) - np.array(ball_curr['box'][:2])), 
                                 default=None)
            
            if closest_player:
                current_possession = closest_player['box'][:2]
                if ball_possession is None or np.linalg.norm(np.array(current_possession) - np.array(ball_possession)) > 2:
                    if ball_possession is not None:
                        events.append({'type': 'possession_change', 'frame': curr_frame['frame']})
                        logger.info(f"Possession change detected at frame {curr_frame['frame']}")
                    ball_possession = current_possession
        
        # Detect player collisions (potential fouls or tackles)
        players = [d for d in curr_frame['detections'] if d['type'] == 'person']
        for i, player1 in enumerate(players):
            for player2 in players[i+1:]:
                distance = np.linalg.norm(np.array(player1['box'][:2]) - np.array(player2['box'][:2]))
                if distance < 1:  # Distance threshold in meters
                    events.append({'type': 'possible_tackle', 'frame': curr_frame['frame']})
                    logger.info(f"Possible tackle detected at frame {curr_frame['frame']}")
        
        # Detect if the ball is near the goal (potential goal or save)
        if ball_curr:
            ball_x, ball_y = ball_curr['box'][:2]
            if ball_x < 5 or ball_x > field_width - 5:
                if 30 < ball_y < field_height - 30:
                    events.append({'type': 'possible_goal_or_save', 'frame': curr_frame['frame']})
                    logger.info(f"Possible goal or save detected at frame {curr_frame['frame']}")
    
    # Consolidate events
    consolidated_events = []
    for event in events:
        if not consolidated_events or event['frame'] - consolidated_events[-1]['frame'] > cooldown_frames:
            consolidated_events.append(event)
        elif event['type'] != consolidated_events[-1]['type']:
            consolidated_events.append(event)
    
    logger.info(f"Total events detected: {len(consolidated_events)}")
    return consolidated_events


def iou(box1, box2):
    # Calculate intersection over union
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / float(area1 + area2 - intersection)
    return iou