from ultralytics import YOLO

model = YOLO('models//yolov8_ball_best.pt')
model.to('cuda')
result = model.predict('input_videos/input_video.mp4',conf=0.2, save=True)
print(result)
print("boxes : ")
for box in result[0].boxes:
    print(box)