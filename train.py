from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

# run model with cuda
model.to('cuda')

# Train the model
results = model.train(data='yolov8.yaml', epochs=100, imgsz=640, device=0)