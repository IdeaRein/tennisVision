from ultralytics import YOLO
import cv2

model = YOLO("Models/yolov8m.pt")  # または yolov8x.ptなど精度の良いモデルを指定

video_path = "Datas/tennis_sample.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # conf=0.1で検出（信頼度20%以上のものを取得）
    results = model.predict(frame, conf=0.2)

    annotated_frame = results[0].plot()
    cv2.imshow("Tennis Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
