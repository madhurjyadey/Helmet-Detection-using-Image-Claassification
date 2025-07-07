import ultralytics
from ultralytics import YOLO

model = YOLO("./best.pt")

source = "video.mp4"
results = model.predict(source, imgsz=736, save=True, show=True, show_conf=True, stream=True, conf=0.5, iou=0.2)

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs
