import mss
import numpy as np
import time

def is_flashed(image, threshold=240, flash_percentage=0.8):
    # Convert the image to grayscale
    gray_image = np.mean(image, axis=2)
    
    # Count the number of pixels that are above the threshold
    bright_pixels = np.sum(gray_image > threshold)
    
    # Calculate the percentage of bright pixels
    total_pixels = gray_image.size
    bright_percentage = bright_pixels / total_pixels
    
    # Determine if the screen is flashed
    return bright_percentage > flash_percentage

with mss.mss() as sct:
    # Define the screen area to capture (set to full screen here)
    monitor = sct.monitors[1]

    while True:
        # Capture the screen
        screenshot = np.array(sct.grab(monitor))

        # Check if the screen is flashed
        if is_flashed(screenshot):
            print("Flashed!")
        # Add a small delay to control the loop frequency
        time.sleep(.01)  # 10ms delay
