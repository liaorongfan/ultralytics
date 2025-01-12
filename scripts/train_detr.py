from ultralytics import RTDETR


if __name__ == "__main__":
    # Load a COCO-pretrained RT-DETR-l model
    model = RTDETR("rtdetr-l.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="VOC.yaml", epochs=100, imgsz=640)

    # Run inference with the RT-DETR-l model on the 'bus.jpg' image
    results = model("path/to/bus.jpg")