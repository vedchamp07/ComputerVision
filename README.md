# ComputerVision

This repository contains my computer vision projects implementing various classical CV techniques and algorithms.

## Projects

### Cat vs Dog Classifier

- **Location**: [cat_vs_dog_classifier/](cat_vs_dog_classifier/)
- **Description**: Transfer-learning pipeline using ResNet18 for the Kaggle Dogs vs Cats dataset with automated data split (train/val/test) and evaluation utilities.
- **Key Features**:
  - Data download/organization helper and reproducible 80/10/10 splits
  - Transfer learning with frozen backbone and tuned final layer
  - Training curves and confusion matrix outputs for quick diagnostics
  - Best-model checkpointing (`best_model.pth`) for inference/evaluation

### Circle Detector (Hough Transform)

- **Location**: [circle_detector/](circle_detector/)
- **Description**: Complete implementation of the Day 2 Assignment for detecting circles in images using OpenCV's Hough Circle Transform
- **Key Features**:
  - Robust preprocessing pipeline (grayscale conversion, Gaussian/median blur, histogram equalization)
  - Configurable Hough Circle Transform with full parameter tuning support
  - Professional visualization with green circle outlines, red center markers, and radius labels
  - Comprehensive statistics reporting (count, min/max/avg radius, detailed coordinates)
  - Full CLI interface with extensive parameter options
  - Thoroughly tested on 4 different scenarios: realistic coins, bubble effects, noisy/challenging images, and real-world scenes
  - Meets 100% of assignment requirements with proper error handling and documentation

### Pencil Sketch Converter

- **Location**: [pencil_sketch_converter/](pencil_sketch_converter/)
- **Description**: Artistic image transformation tool that converts regular photos into pencil sketch style images
- **Key Features**:
  - Converts color images to realistic pencil sketch art
  - Applies edge detection and contrast enhancement techniques
  - Supports batch processing of multiple images
  - Flexible output directory management
  - Optimized for various image types and resolutions
  - Includes sample output directory with example results

## Technologies Used

- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computing and array operations
- **Matplotlib** - Visualization and plotting
- **Python 3.14** - Core programming language

## Setup

```bash
# Clone the repository
git clone https://github.com/vedchamp07/ComputerVision.git
cd ComputerVision

# Set up virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies for specific projects

## Cat vs Dog Classifier
cd cat_vs_dog_classifier
pip install -r requirements.txt
# Option A: prepare data via setup script (requires dataset zip available locally)
python setup_data.py
# Train / evaluate
python train.py
python evaluate.py

## Circle Detector
cd circle_detector
pip install -r requirements.txt
python circle_detector.py test_images/test1_mixed_coins.jpg --output-dir results

## Pencil Sketch Converter
cd pencil_sketch_converter
pip install opencv-python numpy
python pencil_sketch.py input_image.jpg
```
