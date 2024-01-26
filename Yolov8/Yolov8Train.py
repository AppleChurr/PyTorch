from ultralytics import YOLO

model = YOLO('yolov8n.pt') # pass any model type
results = model.train(epochs=5)