import argparse
from ultralytics import YOLO

def main(args):
    # Load trained model
    model = YOLO(args.model)

    # Run prediction
    results = model.predict(
        source=args.source,
        imgsz=640,
        conf=args.conf,
        device=0,
        show=args.show,
        save=args.save,
        stream=args.stream
    )

    # IMPORTANT: iterate over results if stream=True
    for r in results:
        pass  # This forces the generator to run

    print("✅ Inference completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")

    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--stream", action="store_true")

    args = parser.parse_args()
    main(args)

