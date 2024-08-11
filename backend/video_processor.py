import cv2
import numpy as np
from ultralytics import YOLO
import logging
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(file_path):
    model = YOLO('yolov8x.pt')
    cap = cv2.VideoCapture(file_path)
    
    frames_processed = 0
    detections = []
    tracked_objects = {}
    next_id = 1
    jersey_colors = []
    team_colors = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        frame_detections = []
        current_objects = {}

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                object_type = model.names[int(cls)]
                
                if object_type in ['person', 'sports ball']:
                    # Try to match with existing tracked objects
                    matched = False
                    for obj_id, obj in tracked_objects.items():
                        if obj['type'] == object_type and iou(box, obj['box']) > 0.3:
                            current_objects[obj_id] = {
                                'type': object_type,
                                'box': box.tolist(),
                                'confidence': float(conf)
                            }
                            if 'team' in obj:
                                current_objects[obj_id]['team'] = obj['team']
                            matched = True
                            break
                    
                    # If no match, create a new tracked object
                    if not matched:
                        new_id = next_id
                        next_id += 1
                        current_objects[new_id] = {
                            'type': object_type,
                            'box': box.tolist(),
                            'confidence': float(conf)
                        }
                        
                        # Extract jersey color
                        if object_type == 'person':
                            jersey_color = extract_jersey_color(frame, box)
                            jersey_colors.append(jersey_color)

        # Perform team assignment if we have enough jersey colors
        if len(jersey_colors) >= 10 and team_colors is None:
            team_colors = KMeans(n_clusters=2, random_state=42).fit(jersey_colors)
        
        # Assign teams to players
        if team_colors is not None:
            for obj_id, obj in current_objects.items():
                if obj['type'] == 'person' and 'team' not in obj:
                    jersey_color = extract_jersey_color(frame, obj['box'])
                    team = team_colors.predict([jersey_color])[0]
                    obj['team'] = int(team)

        tracked_objects = current_objects
        frame_detections = list(current_objects.values())

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

    events = detect_events(detections)

    return {
        'total_frames': frames_processed,
        'detections': detections,
        'events': events,
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

def extract_jersey_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    player_img = frame[y1:y2, x1:x2]
    player_img = cv2.resize(player_img, (50, 50))  # Resize for consistency
    player_img = player_img.reshape((-1, 3))
    player_img = np.float32(player_img)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _, label, center = cv2.kmeans(player_img, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    dominant_color = res.reshape((50, 50, 3))
    
    return dominant_color[0][0]  # Return the most dominant color