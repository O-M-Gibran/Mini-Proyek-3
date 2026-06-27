import argparse
from pathlib import Path

from ultralytics import YOLO


def find_latest_weights() -> Path:
    weight_candidates = []

    root_weights = Path("best.pt")
    if root_weights.exists():
        weight_candidates.append(root_weights)

    weight_candidates.extend(Path(".").glob("**/weights/best.pt"))

    weight_candidates = sorted(
        weight_candidates,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not weight_candidates:
        raise FileNotFoundError(
            "No trained weights found. Provide --weights or train a model first."
        )

    return weight_candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on one image.")
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to model weights. Defaults to the latest runs/**/weights/best.pt.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection.",
    )
    parser.add_argument(
        "--save-dir",
        default="runs/detect/predict",
        help="Directory where prediction results will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image:
        raise SystemExit("Please provide an image path, for example: python test.py path/to/image.jpg")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    weights_path = Path(args.weights) if args.weights else find_latest_weights()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        save=True,
        project=str(Path(args.save_dir).parent),
        name=Path(args.save_dir).name,
    )

    print(f"Loaded weights: {weights_path}")
    print(f"Input image: {image_path}")
    print(f"Saved predictions to: {results[0].save_dir if results else args.save_dir}")


if __name__ == "__main__":
    main()
