from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(data='yolov8.yaml', epochs=50)