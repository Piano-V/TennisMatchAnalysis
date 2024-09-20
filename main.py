from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2

def main():
    #Read Video
    input_video_path = 'input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)

    #Detect Players and Ball
    #NOTE TO REMEMBER : MANAGE SUBS AS TRUE/FALSE DEPENDING ON WHETHER YOU WANT TO READ FROM STUBS OR NOT
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models//yolov8_ball_best.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_pos(ball_detections)
    
    #Court Line Detector
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #Filter Players
    player_detections = player_tracker.choose_players_filter(court_keypoints, player_detections)

    #Mini Court
    mini_court = MiniCourt(video_frames[0])

    #Detect Ball Shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(ball_shot_frames)

    #Drawing Outputs

    #Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    #Draw Court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    #Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    #Draw Frame Numbers
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    save_video(output_video_frames, 'output_videos/output_video.avi')



if __name__ == '__main__':
    main()