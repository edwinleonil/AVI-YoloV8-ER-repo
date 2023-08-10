# use ultralytics package to detect objects in the people.jpg image

from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Run inference on a single image
results = model('people.jpg', save=True)


