import cv2
import numpy as np
from ultralytics import YOLO
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = YOLO('yolov8x.pt')

def process_frame(frame):
    results = model(frame)
    detections = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confs):
            detections.append({
                'type': model.names[int(cls)],
                'box': box.tolist(),
                'confidence': float(conf)
            })
    return detections

async def process_video(file_path, progress_callback=None):
    try:
        cap = cv2.VideoCapture(file_path)
        logger.info(f"Video file opened: {file_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        frames_processed = 0
        detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_detections = process_frame(frame)
            detections.append({
                'frame': frames_processed,
                'detections': frame_detections
            })
            
            frames_processed += 1
            
            if frames_processed % 100 == 0:
                logger.info(f"Processed {frames_processed} frames")
                if progress_callback:
                    progress_callback(frames_processed / total_frames * 100)
        
        cap.release()
        
        logger.info(f"Total frames processed: {frames_processed}")
        
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
    last_event_frame = -1
    cooldown_frames = 25
    ball_possession = None
    possession_team = None
    last_possession_change = -1
    
    field_width = 105
    field_height = 68
    
    def calculate_speed(pos1, pos2, frames_difference):
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        return distance / frames_difference * 24 / 100  # Assuming 24 fps and 100 pixels = 1 meter
    
    for i in range(1, len(detections)):
        prev_frame = detections[i-1]
        curr_frame = detections[i]
        
        ball_prev = next((d for d in prev_frame['detections'] if d['type'] == 'sports ball'), None)
        ball_curr = next((d for d in curr_frame['detections'] if d['type'] == 'sports ball'), None)
        
        if ball_prev and ball_curr:
            ball_speed = calculate_speed(ball_curr['box'][:2], ball_prev['box'][:2], 1)
            
            if ball_speed > 2:  # Lowered threshold for pass detection
                event_type = 'shot' if ball_speed > 6 else 'pass'  # Lowered threshold for shot detection
                if curr_frame['frame'] - last_event_frame > cooldown_frames:
                    events.append({'type': event_type, 'frame': curr_frame['frame'], 'speed': ball_speed, 'team': possession_team or 0})
                    last_event_frame = curr_frame['frame']
            
            closest_player = min((d for d in curr_frame['detections'] if d['type'] == 'person'), 
                                 key=lambda x: np.linalg.norm(np.array(x['box'][:2]) - np.array(ball_curr['box'][:2])), 
                                 default=None)
            
            if closest_player:
                current_possession = closest_player['box'][:2]
                current_team = random.choice([0, 1])  # Randomly assign team for now
                
                if ball_possession is None or np.linalg.norm(np.array(current_possession) - np.array(ball_possession)) > 4:
                    if possession_team is None or (current_team != possession_team and curr_frame['frame'] - last_possession_change > cooldown_frames):
                        events.append({'type': 'possession_change', 'frame': curr_frame['frame'], 'team': current_team})
                        last_possession_change = curr_frame['frame']
                        last_event_frame = curr_frame['frame']
                        possession_team = current_team
                    ball_possession = current_possession
        
        players = [d for d in curr_frame['detections'] if d['type'] == 'person']
        for i, player1 in enumerate(players):
            for player2 in players[i+1:]:
                distance = np.linalg.norm(np.array(player1['box'][:2]) - np.array(player2['box'][:2]))
                if distance < 0.6 and curr_frame['frame'] - last_event_frame > cooldown_frames:
                    event_type = 'player_interaction'
                    events.append({'type': event_type, 'frame': curr_frame['frame']})
                    last_event_frame = curr_frame['frame']
    
    return events

def generate_commentary(events):
    commentary = []
    for event in events:
        if event['type'] == 'possession_change':
            commentary.append(f"Team {event['team']} gains possession at frame {event['frame']}.")
        elif event['type'] == 'pass':
            commentary.append(f"A pass is made by Team {event['team']} at frame {event['frame']} with a speed of {event['speed']:.2f} m/s.")
        elif event['type'] == 'shot':
            commentary.append(f"A powerful shot is taken by Team {event['team']} at frame {event['frame']} with a speed of {event['speed']:.2f} m/s!")
        elif event['type'] == 'player_interaction':
            commentary.append(f"Close player interaction at frame {event['frame']}.")
    return commentary

def calculate_statistics(events, total_frames):
    team_possession = {0: 0, 1: 0}
    passes = {0: 0, 1: 0}
    shots = {0: 0, 1: 0}
    last_possession_change = 0
    current_team = None

    for event in events:
        if event['type'] == 'possession_change':
            if current_team is not None:
                team_possession[current_team] += event['frame'] - last_possession_change
            current_team = event['team']
            last_possession_change = event['frame']
        elif event['type'] == 'pass':
            passes[event['team']] += 1
        elif event['type'] == 'shot':
            shots[event['team']] += 1

    # Add the final possession duration
    if current_team is not None:
        team_possession[current_team] += total_frames - last_possession_change
    
    # If no possession changes were detected, split possession equally
    if sum(team_possession.values()) == 0:
        team_possession = {0: total_frames / 2, 1: total_frames / 2}

    total_possession = sum(team_possession.values())
    possession_percentages = {team: (count / total_possession) * 100 for team, count in team_possession.items()}

    return {
        'possession': possession_percentages,
        'passes': passes,
        'shots': shots
    }