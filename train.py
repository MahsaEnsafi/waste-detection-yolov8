from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="dataset/data.yaml",
        epochs=20,
        imgsz=640,
        device=0,
        workers=0,
        # Augmentation
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        mosaic=1.0,
        mixup=0.1
    )

if __name__ == "__main__":
    main()
