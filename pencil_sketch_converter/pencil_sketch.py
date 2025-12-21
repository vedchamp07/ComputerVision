import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse

def pencil_sketch(image_path, blur_kernel=21, color=False):
    """
    Convert an image to pencil sketch effect.
    
    Args:
        image_path (str): Path to input image
        blur_kernel (int): Gaussian blur kernel size (must be odd)
        color (bool): Whether to generate a color pencil sketch
    Returns:
        tuple: (original_rgb, sketch) or (None, None) if error
    """
    # Step 1: Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found. Check the path.")

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Invert grayscale
    inverted = 255 - gray

    # Step 4: Apply Gaussian blur
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)

    # Step 5: Invert blurred image
    inverted_blur = 255 - blurred

    # Step 6: Divide and scale (dodge blend)
    inverted_blur = inverted_blur.astype(np.float32)
    gray_f = gray.astype(np.float32)
    sketch_gray = gray_f / (inverted_blur + 1e-6) * 256.0
    sketch_gray = np.clip(sketch_gray, 0, 255).astype(np.uint8)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not color:
        return original_rgb, sketch_gray

    # ----- Color pencil sketch -----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)

    # Slight desaturation for realism
    s = (s * 0.6).astype(np.uint8)

    color_sketch_hsv = cv2.merge([h, s, sketch_gray])
    color_sketch = cv2.cvtColor(color_sketch_hsv, cv2.COLOR_HSV2RGB)

    return original_rgb, color_sketch


def display_result(original, sketch, save_path=None):
    """
    Display original and sketch side-by-side.
    
    Args:
        original: Original image (RGB)
        sketch: Sketch image (grayscale or color)
        save_path: Optional path to save the sketch
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    # Handle grayscale or color sketches
    if len(sketch.shape) == 2:
        axes[1].imshow(sketch, cmap='gray')
    else:
        axes[1].imshow(sketch)
    axes[1].set_title("Pencil Sketch Effect", fontweight="bold")
    axes[1].axis("off")
    
    plt.tight_layout()
    
    # Save the sketch if path provided
    if save_path:
        # Convert RGB back to BGR for OpenCV saving
        if len(sketch.shape) == 2:
            cv2.imwrite(save_path, sketch)
        else:
            sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, sketch_bgr)
        print(f"Sketch saved to: {save_path}")
    
    plt.show()
    
def process_video(video_path, output_path, blur_kernel=21):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Video file not found or cannot be opened.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (blur_kernel | 1, blur_kernel | 1), 0)
        inverted_blur = 255 - blurred

        sketch = gray.astype(np.float32) / (inverted_blur.astype(np.float32) + 1e-6) * 256.0
        sketch = np.clip(sketch, 0, 255).astype(np.uint8)
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        out.write(sketch_bgr)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Update every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')

    print(f"\nProcessing: 100.0% ({total_frames}/{total_frames} frames)")
    cap.release()
    out.release()

def is_valid_save_path(filepath):
    path = Path(filepath)
    
    # 1. Check if the parent directory exists
    if not path.parent.exists():
        print(f"Error: The directory '{path.parent}' does not exist.")
        return False
    
    # 2. Check if the directory is actually a directory
    if not path.parent.is_dir():
        print(f"Error: '{path.parent}' is not a directory.")
        return False

    # 3. Check for write permissions
    # We check the parent folder because the file itself might not exist yet
    if not os.access(path.parent, os.W_OK):
        print(f"Error: You do not have write permissions for '{path.parent}'.")
        return False

    return True

def is_valid_path(filepath):
    path = Path(filepath)
    
    # 1. Check if the parent directory exists
    if not path.parent.exists():
        print(f"Error: The directory '{path.parent}' does not exist.")
        return False
    
    # 2. Check if the directory is actually a directory
    if not path.parent.is_dir():
        print(f"Error: '{path.parent}' is not a directory.")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Pencil Sketch Converter")
    parser.add_argument("input", help="Path to input image or video")
    parser.add_argument("--save", help="Path to save output image or video")
    parser.add_argument("--blur", type=int, default=21, help="Gaussian blur kernel size (odd)")
    parser.add_argument("--color", action="store_true", help="Enable color pencil sketch")
    parser.add_argument("--video", action="store_true", help="Process input as video")

    args = parser.parse_args()

    try:
        if args.video:
            if not args.save:
                raise ValueError("Output path required for video processing.")
            process_video(args.input, args.save, blur_kernel=args.blur)
            print("✓ Video processing complete.")
            return

        if not is_valid_path(args.input):
            return

        original, sketch = pencil_sketch(
            image_path=args.input,
            blur_kernel=args.blur,
            color=args.color
        )

        # Always display, only save if valid path provided
        save_path = None
        if args.save:
            if is_valid_save_path(args.save):
                save_path = args.save
            else:
                print("Warning: Invalid save path. Image will be displayed but not saved.")
        
        display_result(original, sketch, save_path)

    except Exception as e:
        print(f"❌ Processing failed: {e}")

if __name__ == '__main__':
    main()