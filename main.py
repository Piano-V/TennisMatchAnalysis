import gradio as gr
from utils import read_video, save_video, measure_distance, convert_meters_to_pixel_distance, convert_pixel_distance_to_meters, draw_player_stats
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from copy import deepcopy
import constants
import cv2
import pandas as pd
import numpy as np
import os

# Adapted main function for Gradio
def run_analysis(input_video, output_filename, use_stubs):
    # Get the path of the uploaded video
    input_video_path = input_video.name if isinstance(input_video, gr.File) else input_video

    # Read the video
    video_frames = read_video(input_video_path)

    # Ensure output directory exists
    output_dir = 'output_videos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolov8_ball_best.pt")

    # Option to read from stubs
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=use_stubs, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=use_stubs, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_pos(ball_detections)
    
    # Court Line Detector
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Filter Players
    player_detections = player_tracker.choose_players_filter(court_keypoints, player_detections)

    # Mini Court
    mini_court = MiniCourt(video_frames[0])

    # Detect Ball Shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Converting Pos to Mini-Court Pos
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)

    # Player Stats Data
    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_average_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_average_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]
    
    # Calculate stats for each shot
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # Assuming 24 fps

        # Ball distance and speed calculation
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # Player who shot the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id], ball_mini_court_detections[start_frame][1]))

        # Opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id], player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())
        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        # Update player stats
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        # Update averages
        current_player_stats[f'player_{player_shot_ball}_average_shot_speed'] = (
            current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] /
            current_player_stats[f'player_{player_shot_ball}_number_of_shots']
            if current_player_stats[f'player_{player_shot_ball}_number_of_shots'] > 0 else 0
        )
        current_player_stats[f'player_{opponent_player_id}_average_player_speed'] = (
            current_player_stats[f'player_{opponent_player_id}_total_player_speed'] /
            (ball_shot_ind + 1)  # or keep track of opponent shots separately
        )

        player_stats_data.append(current_player_stats)

    # Dataframe for player stats
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # Drawing outputs
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(120, 255, 255))
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Draw frame numbers
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save the final video
    full_output_path = f'{output_dir}/{output_filename}.avi'
    print(f'Saving video to: {full_output_path}')
    print(f'Number of frames: {len(output_video_frames)}')
    save_video(output_video_frames, full_output_path)
    
    # Return both the success message and the video path
    return f'Output video saved as {full_output_path}', full_output_path


# Gradio Interface
input_video = gr.File(label="Upload Video", file_types=[".mp4", ".avi"])
output_file = gr.Textbox(label="Output File Name", value="output_video")
use_stubs = gr.Checkbox(label="Use Stored Stubs", value=False)

# Update the outputs to include video
gr.Interface(fn=run_analysis, inputs=[input_video, output_file, use_stubs], outputs=[gr.Textbox()]).launch()

