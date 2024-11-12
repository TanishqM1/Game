import cv2
import numpy as np
import pytesseract
from mss import mss

# Define the screen region to capture (x, y, width, height)
monitor = {"top": 1007, "left": 616, "width": 71, "height": 54}
deathCam = {"top": 1014, "left": 1112, "width": 68, "height": 18}
weapons = {"CZ75-Auto", "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18", "P2000", "P250", "R8 Revolver", "Tec-9", "USP-S", "AK-47",  
           "AUG", "AWP", "FAMAS", "G3SG1", "Galil AR", "M4A1-S", "M4A4", "SCAR-20", "SG 553", "SSG 08", "MAC-10", "MP5-SD", "MP7", "MP9", "PP-Bizon", 
            "P90", "P9o", "UMP-45", "MAG-7", "Nova", "Sawed-Off", "XM1014", "M249", "Negev"}

def detect():
    with mss() as sct:

        # Capture the screen
        img = np.array(sct.grab(monitor))
        imgD = np.array(sct.grab(deathCam))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayD = cv2.cvtColor(imgD, cv2.COLOR_BGR2GRAY)

        # Apply OCR to the HP area
        hp_text = pytesseract.image_to_string(gray, config="--psm 6")
        death_text = pytesseract.image_to_string(grayD, config="--psm 6")
        try:
            hp = int(hp_text.strip())
            if death_text.strip() in weapons:
                return -2 #dead
            if hp == 100 and death_text =="":
                return -1 #alive
            return hp
        except Exception:
            return 101      