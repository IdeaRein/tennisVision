from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path = "tennis_sample.mp4"  # 検出したい動画ファイルの名前（同じフォルダに置く）

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Tennis Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()