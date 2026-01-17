# Circle Detector (Hough Transform)

## Overview

A simple CLI tool to detect circles in images using OpenCV's Hough Circle Transform. It loads an image, preprocesses it, detects circles, annotates the result, computes summary stats, and saves outputs.

## Quickstart

1. **Install dependencies** (from this folder):

   ```bash
   pip install -r requirements.txt
   ```

2. **Run detection** on an image:

   ```bash
   python circle_detector.py test_images/your_image.jpg --output-dir results --show
   ```

3. **Outputs**:
   - Annotated image: `results/<input>_annotated.jpg`
   - Stats report: `results/statistics.txt`

## Arguments

- `image` (positional): path to input image
- `--output-dir`: folder to save outputs (default: `results`)
- `--dp`: inverse accumulator ratio (default: 1.2)
- `--min-dist`: minimum distance between circle centers (default: 50)
- `--param1`: upper Canny threshold (default: 100)
- `--param2`: accumulator threshold; lower = more circles (default: 30)
- `--min-radius` / `--max-radius`: radius bounds; max 0 means auto (default: 10 / 0)
- `--blur-ksize`: blur kernel size (odd, default: 5)
- `--median-blur`: use median blur instead of Gaussian
- `--equalize-hist`: apply histogram equalization to grayscale
- `--show`: display original vs annotated in a matplotlib window
- `--save-name`: custom filename for annotated image

## Suggested Test Images

Place at least three images under `test_images/`:

- Coins or simple circular objects
- Overlapping circles
- Noisy / varied-size circles

## Parameter Tuning Tips

- **If circles are missed**: try lowering `--param2` or `--min-dist`
- **If false positives appear**: raise `--param2` or increase `--min-dist`
- **Speed vs accuracy**: adjust `--dp` upward (e.g., 1.5) for speed, downward (e.g., 1.1) for accuracy
- **Noise handling**: median helps with salt-and-pepper noise; Gaussian is a good default

## Example Commands

```bash
# Basic usage
python circle_detector.py test_images/test1_mixed_coins.jpg

# With parameter tuning for overlapping circles
python circle_detector.py test_images/test2_bubbles.jpg --param2 25 --min-dist 40

# For noisy images
python circle_detector.py test_images/test3_challenging.jpg --median-blur --param2 35 --min-dist 60

# Display results interactively
python circle_detector.py test_images/test1_mixed_coins.jpg --show
```

## Assignment Requirements Met

✅ **Functional Requirements**:

- Load and preprocess image (grayscale, blur, optional histogram equalization)
- Detect circles using `cv2.HoughCircles` with parameter tuning support
- Visualize results (green outlines, red centers, labeled with ID and radius)
- Report statistics (count, min/max/avg radius, detailed list)
- Save annotated image and statistics file

✅ **Code Quality**:

- Multiple functions with docstrings
- Parameter validation and error handling
- PEP 8 naming conventions
- Command-line interface for flexibility

✅ **Testing**:

- Four test images covering different scenarios
- Verified detection results with various parameter combinations
