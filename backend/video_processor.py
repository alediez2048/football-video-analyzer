import cv2
import numpy as np
from ultralytics import YOLO
import logging
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(file_path):
    try:
        model = YOLO('yolov8x.pt')
        logger.info(f"Model loaded successfully. Class names: {model.names}")
        
        cap = cv2.VideoCapture(file_path)
        logger.info(f"Video file opened: {file_path}")
        
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
            logger.info(f"Frame {frames_processed}: {len(results[0].boxes)} detections")

            frame_detections = []
            current_objects = {}

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for box, cls, conf in zip(boxes, classes, confs):
                    object_type = model.names[int(cls)]
                    logger.info(f"Detected: {object_type} with confidence {conf}")
                    
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
        commentary = generate_commentary(events)
        statistics = calculate_statistics(events, frames_processed)

        return {
            'total_frames': frames_processed,
            'detections': detections,
            'events': events,
            'commentary': commentary,
            'statistics': statistics,
            'field_dimensions': {'width': 105, 'height': 68}  # in meters
        }
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}", exc_info=True)
        raise

def detect_events(detections):
    events = []
    ball_possession = None
    possession_team = None
    ball_prev = None
    last_event_frame = -5  # Reduced cooldown period for short clips
    
    for i in range(1, len(detections)):
        curr_frame = detections[i]
        frame_number = curr_frame['frame']

        if frame_number - last_event_frame < 5:  # Reduced skip period
            continue

        ball_curr = next((d for d in curr_frame['detections'] if d['type'] == 'sports ball'), None)
        players = [d for d in curr_frame['detections'] if d['type'] == 'person']

        if ball_curr and ball_prev:
            ball_speed = calculate_speed(ball_curr['box'][:2], ball_prev['box'][:2], 1)
            closest_player = min(players, key=lambda x: np.linalg.norm(np.array(x['box'][:2]) - np.array(ball_curr['box'][:2])), default=None)
            
            if closest_player:
                current_possession = closest_player['box'][:2]
                current_team = closest_player.get('team')
                
                if ball_possession is None or np.linalg.norm(np.array(current_possession) - np.array(ball_possession)) > 20:
                    if ball_possession is not None and current_team != possession_team:
                        events.append({'type': 'possession_change', 'frame': frame_number, 'team': current_team})
                        last_event_frame = frame_number
                    ball_possession = current_possession
                    possession_team = current_team

                if 0.5 < ball_speed <= 5:  # Adjusted speed thresholds for passes
                    events.append({'type': 'pass', 'frame': frame_number, 'speed': ball_speed, 'team': possession_team})
                    last_event_frame = frame_number
                elif ball_speed > 5:  # Adjusted speed threshold for shots
                    events.append({'type': 'shot', 'frame': frame_number, 'speed': min(ball_speed, 30)})  # Cap shot speed at 30 m/s
                    last_event_frame = frame_number

        ball_prev = ball_curr

    return events

def generate_commentary(events):
    commentary = []
    for event in events:
        if event['type'] == 'possession_change':
            commentary.append(f"Team {event['team']} gains possession at frame {event['frame']}.")
        elif event['type'] == 'pass':
            commentary.append(f"A pass is made at frame {event['frame']} with a speed of {event['speed']:.2f} m/s.")
        elif event['type'] == 'shot':
            commentary.append(f"A powerful shot is taken at frame {event['frame']} with a speed of {event['speed']:.2f} m/s!")
    return commentary

def calculate_statistics(events, total_frames):
    team_possession = {0: 0, 1: 0}
    passes = {0: 0, 1: 0}
    shots = {0: 0, 1: 0}
    last_possession_change = 0
    current_team = 0  # Start with team 0 having possession

    for event in events:
        if event['type'] == 'possession_change':
            team_possession[current_team] += event['frame'] - last_possession_change
            current_team = event['team']
            last_possession_change = event['frame']
        elif event['type'] == 'pass':
            passes[event['team']] += 1
        elif event['type'] == 'shot':
            shots[event['team']] += 1

    # Add the final possession duration
    team_possession[current_team] += total_frames - last_possession_change

    total_possession = sum(team_possession.values())
    
    # Handle the case where there are no possession changes
    if total_possession == 0:
        possession_percentages = {0: 50, 1: 50}  # Default to 50-50 if no possession data
    else:
        possession_percentages = {team: (count / total_possession) * 100 for team, count in team_possession.items()}

    return {
        'possession': possession_percentages,
        'passes': passes,
        'shots': shots
    }

def calculate_speed(pos1, pos2, frames_difference):
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    return distance / frames_difference * 24 / 200  # Assuming 24 fps and 200 pixels = 1 meter

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