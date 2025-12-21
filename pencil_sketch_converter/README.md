# Pencil Sketch Converter

A Python program that transforms photographs into realistic pencil sketch drawings using classical image processing techniques. This project implements the "dodge and burn" method adapted for digital image processing.

## Features

- **Basic Pencil Sketch**: Convert any image to a grayscale pencil sketch
- **Color Pencil Sketch**: Generate colored sketch versions with realistic desaturation
- **Video Processing**: Apply pencil sketch effect to videos frame-by-frame
- **Adjustable Blur**: Customize the blur kernel size for different artistic effects
- **Command-line Interface**: Easy-to-use CLI with multiple options

## Requirements

```bash
pip install opencv-python numpy matplotlib
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install opencv-python numpy matplotlib
   ```

## Usage

### Basic Usage

Convert an image to pencil sketch:
```bash
python pencil_sketch.py input_image.jpg
```

### Save Output

Save the sketch to a file:
```bash
python pencil_sketch.py input_image.jpg --save output_sketch.jpg
```

### Adjust Blur Kernel

Change the blur kernel size (must be odd number):
```bash
python pencil_sketch.py input_image.jpg --blur 31
```

### Color Pencil Sketch

Generate a colored sketch version:
```bash
python pencil_sketch.py input_image.jpg --color --save color_sketch.jpg
```

### Video Processing

Process a video file:
```bash
python pencil_sketch.py input_video.mp4 --video --save output_video.mp4
```

### Combined Options

```bash
python pencil_sketch.py portrait.jpg --blur 25 --color --save colored_portrait.jpg
```

## Command-Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `input` | Path to input image or video | Yes | - |
| `--save` | Path to save output | No | - |
| `--blur` | Gaussian blur kernel size (odd) | No | 21 |
| `--color` | Enable color pencil sketch | No | False |
| `--video` | Process input as video | No | False |

## Algorithm Overview

The pencil sketch effect is created through the following steps:

1. **Grayscale Conversion**: Convert the color image to grayscale
2. **Inversion**: Create a negative by inverting pixel values
3. **Gaussian Blur**: Apply blur with large kernel (default 21x21) for soft tones
4. **Second Inversion**: Invert the blurred image
5. **Dodge Blend**: Divide grayscale by inverted blur and scale to create the sketch effect

The mathematical formula for the final step:
```
sketch(x, y) = min(255, (gray(x, y) / inverted_blur(x, y)) × 256)
```

## Testing

The program has been tested with various image types:

- **Portraits/Faces**: Best results with blur kernel 21-25
- **Landscapes/Scenery**: Works well with blur kernel 25-31
- **Objects with Distinct Edges**: Sharp results with blur kernel 15-21

## File Structure

```
pencil_sketch_converter/
├── pencil_sketch.py       # Main program
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── test_files/            # Sample input images
│   ├── bird.mp4
│   ├── cat.jpg
│   ├── cube.jpg
|   └── landscape.jpg
└── output_sketches/       # Generated sketches
    ├── bird_sketch.mp4
    ├── cat_sketch.jpg
    ├── cube_sketch.jpg
    └── landscape_color.jpg
```

## Error Handling

The program includes robust error handling for:
- File not found errors
- Invalid image formats
- Invalid directory paths
- Write permission issues
- Processing errors (division by zero, overflow)

## Observations

### Best Practices
- **Blur kernel size**: Larger kernels (25-31) create softer, more artistic sketches
- **Image quality**: High-resolution images produce better detail
- **Subject matter**: Images with clear edges and good contrast work best
- **Color sketches**: Slight desaturation (60%) provides more realistic results

### Performance Notes
- Image processing is near-instantaneous for standard photos
- Video processing time depends on video length and resolution
- Large blur kernels slightly increase processing time

## Challenges Faced

1. **Division by Zero**: Solved by adding small epsilon (1e-6) to denominator
2. **Data Type Handling**: Ensured proper float32 conversion before division and uint8 for output
3. **BGR vs RGB**: Handled OpenCV's BGR format and converted to RGB for display
4. **Video Frame Rate**: Maintained original video FPS in output
5. **Color Sketch Realism**: Achieved natural look through HSV color space and desaturation

## Bonus Features Implemented

- **Adjustable Blur Parameter**: Command-line argument for blur kernel size
- **Color Pencil Sketch**: HSV-based colored sketch with desaturation
- **Video Processing**: Frame-by-frame video sketch conversion

## Examples

### Portrait
```bash
python pencil_sketch.py portrait.jpg --blur 21 --save portrait_sketch.jpg
```
Creates a classic pencil sketch with fine details.

### Landscape
```bash
python pencil_sketch.py landscape.jpg --blur 31 --color --save landscape_color.jpg
```
Generates a softer, colored artistic sketch.

### Video
```bash
python pencil_sketch.py demo.mp4 --video --blur 21 --save demo_sketch.mp4
```
Converts entire video to pencil sketch effect.

## Tips for Best Results

- Start with blur kernel 21 and adjust based on desired effect
- Use `--color` flag for more vibrant, artistic results
- Higher blur values (25-31) work better for artistic/abstract looks
- Lower blur values (15-21) preserve more detail for technical drawings
- Ensure even kernel sizes are automatically corrected to odd values

## License

This project is created for educational purposes.

## Author

Vedant Narayanswami

## Acknowledgments

- Algorithm based on the "dodge and burn" photography technique
- OpenCV for image processing capabilities
- NumPy for efficient numerical operations