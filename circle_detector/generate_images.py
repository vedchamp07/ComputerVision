"""Generate additional test images with more realistic scenarios."""

import cv2
import numpy as np
from pathlib import Path


def generate_mixed_coins_image(output_path: str) -> None:
    """Generate realistic coin-like objects with different textures."""
    image = np.ones((500, 600, 3), dtype=np.uint8) * 245
    
    # Add subtle background texture
    noise = np.random.normal(0, 8, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw coins with different colors and subtle gradients
    # Large coin - golden
    cv2.circle(image, (120, 120), 55, (50, 180, 220), -1)
    cv2.circle(image, (120, 120), 50, (80, 200, 240), -1)
    
    # Medium coin - silver
    cv2.circle(image, (300, 100), 40, (160, 160, 160), -1)
    cv2.circle(image, (300, 100), 35, (180, 180, 180), -1)
    
    # Small coin - copper
    cv2.circle(image, (450, 150), 30, (100, 140, 180), -1)
    cv2.circle(image, (450, 150), 25, (120, 160, 200), -1)
    
    # Overlapping coins
    cv2.circle(image, (200, 300), 45, (90, 170, 200), -1)
    cv2.circle(image, (240, 320), 42, (110, 190, 220), -1)
    
    # Partial coin at edge
    cv2.circle(image, (550, 400), 38, (70, 150, 190), -1)
    
    # Various sizes
    cv2.circle(image, (100, 400), 35, (140, 120, 100), -1)
    cv2.circle(image, (350, 350), 50, (60, 140, 180), -1)
    cv2.circle(image, (480, 300), 25, (130, 170, 200), -1)
    
    cv2.imwrite(output_path, image)
    print(f"Generated: {output_path}")


def generate_bubbles_image(output_path: str) -> None:
    """Generate bubble-like circles with transparency effects."""
    image = np.ones((450, 550, 3), dtype=np.uint8) * 220
    
    # Create gradient background
    for i in range(image.shape[0]):
        factor = i / image.shape[0]
        image[i, :] = image[i, :] * (0.8 + 0.4 * factor)
    
    # Bubble circles with different transparency effects
    circles_data = [
        (100, 80, 40, (200, 220, 255)),
        (250, 120, 35, (180, 255, 200)),
        (400, 100, 45, (255, 200, 180)),
        (150, 250, 50, (220, 180, 255)),
        (350, 200, 30, (255, 255, 180)),
        (80, 350, 38, (180, 255, 255)),
        (280, 320, 42, (255, 180, 220)),
        (450, 300, 35, (200, 255, 180)),
        (500, 400, 28, (255, 200, 200)),
    ]
    
    for x, y, r, color in circles_data:
        # Outer ring (darker)
        cv2.circle(image, (x, y), r, tuple(int(c * 0.7) for c in color), 2)
        # Inner fill (lighter)
        cv2.circle(image, (x, y), r-3, color, -1)
        # Highlight
        cv2.circle(image, (x-10, y-10), 8, (255, 255, 255), -1)
    
    cv2.imwrite(output_path, image)
    print(f"Generated: {output_path}")


def generate_challenging_image(output_path: str) -> None:
    """Generate challenging image with partial circles, noise, and varying contrast."""
    image = np.random.randint(200, 240, (400, 500, 3), dtype=np.uint8)
    
    # Add structured noise
    for i in range(0, image.shape[0], 20):
        for j in range(0, image.shape[1], 25):
            cv2.rectangle(image, (j, i), (j+5, i+5), 
                         (np.random.randint(150, 200), 
                          np.random.randint(150, 200), 
                          np.random.randint(150, 200)), -1)
    
    # Perfect circles (should be detected)
    cv2.circle(image, (100, 100), 35, (80, 80, 80), -1)
    cv2.circle(image, (300, 80), 28, (70, 70, 70), -1)
    cv2.circle(image, (420, 150), 40, (90, 90, 90), -1)
    
    # Partial circles (at edges - challenging)
    cv2.circle(image, (0, 200), 45, (60, 60, 60), -1)
    cv2.circle(image, (500, 300), 38, (85, 85, 85), -1)
    
    # Circles with poor contrast
    cv2.circle(image, (200, 250), 32, (180, 180, 180), -1)
    cv2.circle(image, (350, 300), 25, (190, 190, 190), -1)
    
    # Ellipse (should not be detected as circle)
    cv2.ellipse(image, (250, 350), (40, 25), 0, 0, 360, (100, 100, 100), -1)
    
    # Add more noise on top
    noise = np.random.normal(0, 20, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, image)
    print(f"Generated: {output_path}")


def generate_real_world_scene(output_path: str) -> None:
    """Generate a more realistic scene with circular objects."""
    # Create a table-like background
    image = np.ones((450, 600, 3), dtype=np.uint8) * 200
    
    # Wood grain texture
    for i in range(image.shape[0]):
        wave = int(10 * np.sin(i * 0.05))
        image[i, :] = np.clip(image[i, :].astype(np.int16) + wave, 0, 255).astype(np.uint8)
    
    # Plate (large circle)
    cv2.circle(image, (300, 225), 120, (240, 240, 240), -1)
    cv2.circle(image, (300, 225), 118, (220, 220, 220), 2)
    
    # Coins on the plate
    cv2.circle(image, (250, 200), 25, (180, 140, 100), -1)  # Penny
    cv2.circle(image, (350, 190), 22, (200, 200, 200), -1)  # Nickel
    cv2.circle(image, (280, 250), 20, (220, 180, 120), -1)  # Dime
    cv2.circle(image, (320, 260), 27, (190, 150, 110), -1)  # Quarter
    
    # Cup (circular rim)
    cv2.circle(image, (500, 150), 45, (150, 100, 80), 3)
    cv2.circle(image, (500, 150), 42, (180, 130, 100), -1)
    
    # Buttons
    cv2.circle(image, (100, 100), 18, (50, 50, 150), -1)
    cv2.circle(image, (150, 120), 16, (150, 50, 50), -1)
    cv2.circle(image, (80, 350), 20, (50, 150, 50), -1)
    
    cv2.imwrite(output_path, image)
    print(f"Generated: {output_path}")


def main() -> None:
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    generate_mixed_coins_image(str(test_dir / "test4_mixed_coins.jpg"))
    generate_bubbles_image(str(test_dir / "test5_bubbles.jpg"))
    generate_challenging_image(str(test_dir / "test6_challenging.jpg"))
    generate_real_world_scene(str(test_dir / "test7_real_scene.jpg"))
    
    print("All additional test images generated successfully!")


if __name__ == "__main__":
    main()