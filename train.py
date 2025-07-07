from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('yolo11s.pt')
    model.train(data="./data.yaml", epochs=100, imgsz=416, batch=16, device="0", mixup=0.5)
