from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="VOC.yaml", epochs=100, imgsz=640)