from ultralytics import YOLO
from pascal_voc_writer import Writer
import glob
import os


if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11l.pt")  # load an official model
    model = YOLO("runs/detect/train3/weights/best.pt")  # load a custom model
    imgs = glob.glob("/root/datasets/omni/images/test_front/*.JPG")
    for img in imgs:
        # Predict with the model
        results = model(img)
        h, w = results[0].orig_shape
        # w_scale, h_scale = w / 960, h / 640
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        writer = Writer(img, w, h)
        for cls, bbox in zip (labels, bboxes):
            x1, y1, x2, y2 = bbox
            # x1, y1, x2, y2 = x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale
            writer.addObject(str(cls), x1, y1, x2, y2)
        label_dir = "/root/front/pred/test_front"
        os.makedirs(label_dir, exist_ok=True)
        save_to = os.path.join(label_dir, os.path.basename(img).replace(".JPG", ".xml"))
        writer.save(save_to)
        results[0].save("tmp.jpg")
        
