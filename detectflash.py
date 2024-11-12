import mss
import numpy as np
import time

def is_flashed(image, threshold=240, flash_percentage=0.8):
    # Convert the image to grayscale
    gray_image = np.mean(image, axis=2)
    
    # Count number of pixels considered "bright" (over our threshold)
    bright_pixels = np.sum(gray_image > threshold)
    
    # Calculate the percentage of bright pixels
    total_pixels = gray_image.size
    bright_percentage = bright_pixels / total_pixels
    
    # Determine if we are flashed.
    return bright_percentage > flash_percentage

with mss.mss() as sct:
    #Capture Full-Screen
    monitor = sct.monitors[1]

    while True:
        screenshot = np.array(sct.grab(monitor))

        #Pass our screenshot to is_flashed
        if is_flashed(screenshot):
            print("Flashed!")
        #Check every 1 second (to stop a flash from counting many times)
        time.sleep(.05) #50ms delay.  
