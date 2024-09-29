import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    # Check if there are frames to save
    if not output_video_frames:
        print("No frames to save.")
        return

    # Define the codec and create VideoWriter object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    # Get frame dimensions
    height, width = output_video_frames[0].shape[:2]
    
    # Initialize VideoWriter
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return

    for i, frame in enumerate(output_video_frames):
        out.write(frame)

    out.release()  # Finalize the video file
    print(f"Video saved successfully at {output_video_path}")