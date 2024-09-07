import cv2
import numpy as np
import pytesseract
from mss import mss

# Define the screen region to capture (x, y, width, height)
monitor = {"top": 1007, "left": 616, "width": 71, "height": 54}

with mss() as sct:
    while True:
        # Capture the screen
        img = np.array(sct.grab(monitor))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply OCR to the HP area
        hp_text = pytesseract.image_to_string(gray, config="--psm 6")

        try:
            hp = int(hp_text.strip())
            print(f"HP: {hp}")
        except ValueError:
            print("Failed to detect HP")

        # Break the loop with a key press (e.g., 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()