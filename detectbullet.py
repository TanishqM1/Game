import cv2
import numpy as np
import mss

lower = np.array([0, 150, 20])
upper = np.array([10, 255, 255])

screen_width = 1920
screen_height = 1080
edge_thickness = 50

top_edge = {"top": 0, "left": 0, "width": screen_width, "height": edge_thickness}
bottom_edge = {"top": screen_height - edge_thickness, "left": 0, "width": screen_width, "height": edge_thickness}
left_edge = {"top": 0, "left": 0, "width": edge_thickness, "height": screen_height}
right_edge = {"top": 0, "left": screen_width - edge_thickness, "width": edge_thickness, "height": screen_height}

def detect_red_in_bgr(roi):
    with mss.mss() as sct:
        frame = np.array(sct.grab(roi))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(frame, lower, upper)

        cv2.imshow("Red Detection (BGR)", mask)

        return cv2.countNonZero(mask)
    
while True:

    top_red_pixels = detect_red_in_bgr(top_edge)
    bottom_red_pixels = detect_red_in_bgr(bottom_edge)
    left_red_pixels = detect_red_in_bgr(left_edge)
    right_red_pixels = detect_red_in_bgr(right_edge)

    if top_red_pixels > 500:
        print("Red patch detected at the top!")
    if bottom_red_pixels > 500:
        print("Red patch detected at the bottom!")
    if left_red_pixels > 500:
        print("Red patch detected on the left!")
    if right_red_pixels > 500:
        print("Red patch detected on the right!")
   
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()