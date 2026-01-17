"""
Circle detection assignment solution using Hough Circle Transform.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(
    image_path: str,
    blur_ksize: int = 5,
    use_median: bool = False,
    equalize_hist: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load image, convert to grayscale, and apply smoothing."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Image path does not exist or is not a file: {path}")

    color = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"Failed to read image: {path}")

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    ksize = max(3, blur_ksize | 1)
    if use_median:
        gray = cv2.medianBlur(gray, ksize)
    else:
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    if equalize_hist:
        gray = cv2.equalizeHist(gray)

    return color, gray


def detect_circles(
    gray_image: np.ndarray,
    dp: float = 1.2,
    min_dist: float = 50,
    param1: float = 100,
    param2: float = 30,
    min_radius: int = 10,
    max_radius: int = 0,
) -> Optional[np.ndarray]:
    """Detect circles using Hough Circle Transform."""
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("detect_circles expects a grayscale image")

    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    return np.round(circles[0]).astype(int)


def visualize_circles(
    image: np.ndarray,
    circles: Optional[np.ndarray],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Draw detected circles and optionally display/save the annotated image."""
    annotated = image.copy()

    if circles is not None:
        for idx, (x_center, y_center, radius) in enumerate(circles, start=1):
            cv2.circle(annotated, (x_center, y_center), radius, (0, 255, 0), 2)
            cv2.circle(annotated, (x_center, y_center), 2, (0, 0, 255), 3)
            label = f"ID {idx}: r={radius}px"
            cv2.putText(
                annotated,
                label,
                (x_center - radius, y_center - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)

    if show:
        rgb_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_original)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(rgb_annotated)
        axes[1].set_title("Annotated")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


def calculate_statistics(circles: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """Compute basic statistics for detected circles."""
    if circles is None or len(circles) == 0:
        return {
            "count": 0,
            "min_radius": None,
            "max_radius": None,
            "avg_radius": None,
        }

    radii = circles[:, 2].astype(float)
    return {
        "count": int(len(radii)),
        "min_radius": float(np.min(radii)),
        "max_radius": float(np.max(radii)),
        "avg_radius": float(np.mean(radii)),
    }


def format_statistics(stats: Dict[str, Optional[float]], circles: Optional[np.ndarray]) -> str:
    """Create a human-readable statistics report."""
    lines = ["Circle Detection Statistics"]
    lines.append(f"Total circles: {stats['count']}")

    if stats["count"] == 0:
        lines.append("No circles were detected.")
    else:
        lines.append(f"Min radius: {stats['min_radius']:.2f} px")
        lines.append(f"Max radius: {stats['max_radius']:.2f} px")
        lines.append(f"Avg radius: {stats['avg_radius']:.2f} px")
        lines.append("")
        lines.append("Circles (x, y, r):")
        for idx, (x_center, y_center, radius) in enumerate(circles, start=1):
            lines.append(f"  {idx}: ({x_center}, {y_center}), r={radius} px")

    return "\n".join(lines)


def save_statistics(report: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hough Circle Detector")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs")
    parser.add_argument("--dp", type=float, default=1.2, help="Inverse accumulator ratio")
    parser.add_argument("--min-dist", type=float, default=50.0, help="Min distance between centers")
    parser.add_argument("--param1", type=float, default=100.0, help="Upper Canny threshold")
    parser.add_argument("--param2", type=float, default=30.0, help="Accumulator threshold")
    parser.add_argument("--min-radius", type=int, default=10, help="Minimum circle radius")
    parser.add_argument("--max-radius", type=int, default=0, help="Maximum circle radius (0 for auto)")
    parser.add_argument("--blur-ksize", type=int, default=5, help="Kernel size for blur (odd)")
    parser.add_argument("--median-blur", action="store_true", help="Use median blur instead of Gaussian")
    parser.add_argument("--equalize-hist", action="store_true", help="Apply histogram equalization to gray image")
    parser.add_argument("--show", action="store_true", help="Display original and annotated images")
    parser.add_argument("--save-name", default=None, help="Filename for annotated image (defaults to input name)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    color, gray = preprocess_image(
        args.image,
        blur_ksize=args.blur_ksize,
        use_median=args.median_blur,
        equalize_hist=args.equalize_hist,
    )

    circles = detect_circles(
        gray,
        dp=args.dp,
        min_dist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )

    stats = calculate_statistics(circles)
    report = format_statistics(stats, circles)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_name = Path(args.image).stem
    annotated_name = args.save_name or f"{input_name}_annotated.jpg"
    annotated_path = output_dir / annotated_name
    stats_path = output_dir / "statistics.txt"

    visualize_circles(color, circles, annotated_path, show=args.show)
    save_statistics(report, stats_path)

    print(report)
    print(f"Annotated image saved to: {annotated_path}")
    print(f"Statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
