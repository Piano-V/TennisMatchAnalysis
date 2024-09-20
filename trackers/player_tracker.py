from ultralytics import YOLO
import cv2
import torch
import pickle
import sys
sys.path.append('../')
from utils import measure_distance,get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
    

    def choose_players_filter(self, court_keypoints, player_detections):
        player_detections_one_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_one_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances=[]
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_point = (court_keypoints[i],court_keypoints[i+1])
                distance = measure_distance(player_center, court_point)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id,min_distance))
        
        distances.sort(key=lambda x: x[1])
        #Choose First 2 Tracks
        chosen_players = [x[0] for x in distances[:2]]
        return chosen_players
        

    def detect_frame(self, frame):
        results = self.model.track(frame,persist=True)
        id_name_map = results[0].names
        player_dict = {}
        for box in results[0].boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls = id_name_map[object_cls_id]
            if(object_cls == 'person'):
                player_dict[track_id] = result
        
        return player_dict
    
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections


    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            output_video_frames.append(frame)
        
        return output_video_frames