from ultralytics import YOLO

def main():
    # 前回の学習で得たbest.ptを使って再学習
    model = YOLO('LeaeningModels/best.pt')

    # 再学習の実行
    model.train(
        data='C:/Users/kwhr0/tennisVision/Datasets/tennis_custom/data.yaml',
        epochs=10,      # エポック数は必要に応じて変更
        imgsz=640,      # 画像サイズは必要に応じて変更  
        batch=16,       # 必要に応じて変更
        device='cpu',   # AMDなどCUDA非対応GPUの場合は 'cpu' を指定
    )

if __name__ == '__main__':
    main()
