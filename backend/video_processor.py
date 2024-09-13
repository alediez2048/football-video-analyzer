import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import asyncio
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_video(file_path, progress_callback=None):
    try:
        model = YOLO('yolov8x.pt')
        logger.info(f"Model loaded successfully. Class names: {model.names}")
        
        cap = cv2.VideoCapture(file_path)
        logger.info(f"Video file opened: {file_path}")
        
        frames_processed = 0
        detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)
            frame_detections = process_frame(results, frames_processed)
            
            detections.append(frame_detections)
            
            frames_processed += 1
            
            if frames_processed % 100 == 0:
                logger.info(f"Processed {frames_processed} frames")
                if progress_callback:
                    progress_callback(frames_processed)
        
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

def process_frame(results, frame_number):
    frame_detections = []
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confs):
            object_type = r.names[int(cls)]
            if object_type in ['person', 'sports ball']:
                frame_detections.append({
                    'type': object_type,
                    'box': box.tolist(),
                    'confidence': float(conf)
                })
    
    return {
        'frame': frame_number,
        'detections': frame_detections
    }

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
                    events.append({'type': event_type, 'frame': curr_frame['frame'], 'speed': ball_speed, 'team': possession_team})
                    last_event_frame = curr_frame['frame']
            
            closest_player = min((d for d in curr_frame['detections'] if d['type'] == 'person'), 
                                 key=lambda x: np.linalg.norm(np.array(x['box'][:2]) - np.array(ball_curr['box'][:2])), 
                                 default=None)
            
            if closest_player:
                current_possession = closest_player['box'][:2]
                if 'team' not in closest_player:
                    current_team = random.choice([0, 1])
                else:
                    current_team = closest_player['team']
                
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
                    team1 = player1.get('team', random.choice([0, 1]))
                    team2 = player2.get('team', random.choice([0, 1]))
                    if team1 != team2:
                        event_type = 'tackle'
                    else:
                        event_type = 'pass' if ball_speed > 2 else 'player_interaction'
                    events.append({'type': event_type, 'frame': curr_frame['frame'], 'teams': [team1, team2]})
                    last_event_frame = curr_frame['frame']
        
        if ball_curr:
            ball_x, ball_y = ball_curr['box'][:2]
            if (ball_x < 15 or ball_x > field_width - 15) and 15 < ball_y < field_height - 15:
                if curr_frame['frame'] - last_event_frame > cooldown_frames:
                    events.append({'type': 'possible_goal_or_save', 'frame': curr_frame['frame'], 'team': possession_team})
                    last_event_frame = curr_frame['frame']
        
        # Simple offside detection
        if possession_team is not None:
            defending_team = 1 - possession_team
            defending_players = [p for p in players if p.get('team') == defending_team]
            attacking_players = [p for p in players if p.get('team') == possession_team]
            if defending_players and attacking_players:
                last_defender = max(defending_players, key=lambda p: p['box'][0])
                for attacker in attacking_players:
                    if attacker['box'][0] > last_defender['box'][0] and attacker['box'][0] > ball_curr['box'][0]:
                        events.append({'type': 'possible_offside', 'frame': curr_frame['frame'], 'team': possession_team})
                        break
    
    logger.info(f"Total events detected: {len(events)}")
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
        elif event['type'] == 'tackle':
            commentary.append(f"A tackle occurs between Team {event['teams'][0]} and Team {event['teams'][1]} at frame {event['frame']}.")
        elif event['type'] == 'player_interaction':
            commentary.append(f"Close player interaction at frame {event['frame']}. Possible tackle or challenge.")
        elif event['type'] == 'possible_goal_or_save':
            commentary.append(f"Exciting moment near the goal at frame {event['frame']}! Possible goal or save for Team {event['team']}.")
        elif event['type'] == 'possible_offside':
            commentary.append(f"Possible offside at frame {event['frame']} for Team {event['team']}.")
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
            passes[event.get('team', 0)] += 1
        elif event['type'] == 'shot':
            shots[event.get('team', 0)] += 1

    # Add the final possession duration
    if current_team is not None:
        team_possession[current_team] += total_frames - last_possession_change
    else:
        # If no possession changes were detected, split possession equally
        team_possession = {0: total_frames / 2, 1: total_frames / 2}

    total_possession = sum(team_possession.values())
    possession_percentages = {team: (count / total_possession) * 100 for team, count in team_possession.items()}

    return {
        'possession': possession_percentages,
        'passes': passes,
        'shots': shots
    }

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