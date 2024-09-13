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
        jersey_colors = []
        team_colors = None
        ball_tracker = None
        ball_history = []

        # Parameters for ball tracking
        ball_detector = cv2.SimpleBlobDetector_create()
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        ball_detector = cv2.SimpleBlobDetector_create(params)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, conf=0.3)  # Lower confidence threshold
            logger.info(f"Frame {frames_processed}: {len(results[0].boxes)} detections")

            frame_detections = []
            ball_detected = False

            # Ball detection using SimpleBlobDetector
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints = ball_detector.detect(gray_frame)
            
            if keypoints:
                ball_detected = True
                ball_tracker = (int(keypoints[0].pt[0]), int(keypoints[0].pt[1]))
                ball_history.append(ball_tracker)
                if len(ball_history) > 10:
                    ball_history.pop(0)
                frame_detections.append({
                    'type': 'sports ball',
                    'box': [ball_tracker[0]-5, ball_tracker[1]-5, ball_tracker[0]+5, ball_tracker[1]+5],
                    'id': -1,
                    'confidence': 1.0
                })

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                obj_type = model.names[int(box.cls)]
                track_id = int(box.id) if box.id is not None else -1
                confidence = float(box.conf)
                
                if obj_type == 'person':
                    frame_detections.append({
                        'type': obj_type,
                        'box': [x1, y1, x2, y2],
                        'id': track_id,
                        'confidence': confidence
                    })
                    jersey_color = extract_jersey_color(frame, [x1, y1, x2, y2])
                    jersey_colors.append(jersey_color)

            if not ball_detected and ball_history:
                # Use the average of the last few positions
                avg_x = sum(pos[0] for pos in ball_history) / len(ball_history)
                avg_y = sum(pos[1] for pos in ball_history) / len(ball_history)
                ball_tracker = (int(avg_x), int(avg_y))
                frame_detections.append({
                    'type': 'sports ball',
                    'box': [avg_x-5, avg_y-5, avg_x+5, avg_y+5],
                    'id': -1,
                    'confidence': 0.5
                })

            # Perform team assignment if we have enough jersey colors
            if len(jersey_colors) >= 10 and team_colors is None:
                team_colors = KMeans(n_clusters=2, random_state=42).fit(jersey_colors)
            
            # Assign teams to players
            if team_colors is not None:
                for obj in frame_detections:
                    if obj['type'] == 'person':
                        jersey_color = extract_jersey_color(frame, obj['box'])
                        team = team_colors.predict([jersey_color])[0]
                        obj['team'] = int(team)

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

        # Draw on frame
        cap = cv2.VideoCapture(file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for frame_data in detections:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = draw_on_frame(frame, frame_data['detections'])
            output.write(annotated_frame)
        
        cap.release()
        output.release()

        # Add these lines before calling calculate_statistics
        logger.info(f"Number of events detected: {len(events)}")
        logger.info(f"Events: {events}")

        statistics = calculate_statistics(events, frames_processed)
        return {
            'total_frames': frames_processed,
            'detections': detections,
            'events': events,
            'commentary': commentary,
            'field_dimensions': {'width': 105, 'height': 68},  # in meters
            'statistics': statistics
        }
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'total_frames': frames_processed if 'frames_processed' in locals() else 0,
            'detections': detections if 'detections' in locals() else [],
            'events': [],
            'commentary': [],
            'statistics': {}
        }

def detect_events(detections):
    events = []
    ball_possession = None
    possession_team = None
    ball_prev = None
    last_event_frame = -30  # Cooldown period
    
    for i in range(1, len(detections)):
        curr_frame = detections[i]
        frame_number = curr_frame['frame']

        if frame_number - last_event_frame < 30:  # Skip if within cooldown period
            continue

        ball_curr = next((d for d in curr_frame['detections'] if d['type'] == 'sports ball'), None)
        players = [d for d in curr_frame['detections'] if d['type'] == 'person']

        if ball_curr and ball_prev:
            ball_speed = calculate_speed(ball_curr['box'][:2], ball_prev['box'][:2], 1)
            closest_player = min(players, key=lambda x: np.linalg.norm(np.array(x['box'][:2]) - np.array(ball_curr['box'][:2])), default=None)
            
            if closest_player:
                current_possession = closest_player['box'][:2]
                current_team = closest_player.get('team')
                
                if ball_possession is None or np.linalg.norm(np.array(current_possession) - np.array(ball_possession)) > 50:
                    if ball_possession is not None and current_team != possession_team:
                        events.append({'type': 'possession_change', 'frame': frame_number, 'team': current_team})
                        last_event_frame = frame_number
                    ball_possession = current_possession
                    possession_team = current_team

                if 5 < ball_speed <= 20:  # Adjusted speed thresholds
                    events.append({'type': 'pass', 'frame': frame_number, 'speed': ball_speed, 'team': possession_team})
                    last_event_frame = frame_number
                elif ball_speed > 20:
                    events.append({'type': 'shot', 'frame': frame_number, 'speed': ball_speed, 'team': possession_team})
                    last_event_frame = frame_number

        ball_prev = ball_curr

    return events

def calculate_speed(pos1, pos2, frames_difference):
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    return distance / frames_difference  # Remove the * 24 multiplication

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

def draw_on_frame(frame, detections):
    for obj in detections:
        x1, y1, x2, y2 = map(int, obj['box'])
        if obj['type'] == 'person':
            color = (0, 255, 0) if obj.get('team') == 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Player {obj.get('team', '')}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif obj['type'] == 'sports ball':
            color = (0, 165, 255)  # Orange color for the ball
            center = ((x1+x2)//2, (y1+y2)//2)
            cv2.circle(frame, center, 10, color, -1)  # Draw a filled circle for the ball
            cv2.putText(frame, "Ball", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

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

    for event in events:
        if event['type'] == 'possession_change':
            team_possession[event['team']] += 1
        elif event['type'] == 'pass':
            passes[event['team']] += 1
        elif event['type'] == 'shot':
            shots[event['team']] += 1

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