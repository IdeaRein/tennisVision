import cv2
import os

video_path = "Datas/tennis_sample2.mp4"  # 入力動画
output_dir = "dataset_frames"  # 出力フォルダ
interval = 5  # フレーム間隔（5なら5フレームごと）

# 出力フォルダがなければ作成
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        filename = f"{output_dir}/frame_{saved_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"保存された画像の枚数: {saved_count}")
