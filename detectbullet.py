import cv2
import numpy as np
import mss
from detecthealth import detect

lower = np.array([0, 150, 75])
upper = np.array([7, 255, 255])

screen_width = 1920
screen_height = 1080
edge_thickness = 150

# hp = 100

top_edge = {"top": 0, "left": 0, "width": screen_width, "height": edge_thickness}
bottom_edge = {"top": screen_height - edge_thickness, "left": 0, "width": screen_width, "height": edge_thickness}
left_edge = {"top": 0, "left": 0, "width": edge_thickness, "height": screen_height}
right_edge = {"top": 0, "left": screen_width - edge_thickness, "width": edge_thickness, "height": screen_height}

def detect_red_in_bgr(roi):
    with mss.mss() as sct:
        frame = np.array(sct.grab(roi))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(frame, lower, upper)

        return mask, cv2.countNonZero(mask)
    
def run_detectbullet(hp):
    
    while True:

        top_mask, top_red_pixels = detect_red_in_bgr(top_edge)
        bottom_mask, bottom_red_pixels = detect_red_in_bgr(bottom_edge)
        left_mask, left_red_pixels = detect_red_in_bgr(left_edge)
        right_mask, right_red_pixels = detect_red_in_bgr(right_edge)

        combined_mask = np.zeros((screen_height, screen_width), dtype=np.uint8)
        combined_mask[0:edge_thickness, :] = top_mask
        combined_mask[screen_height-edge_thickness:, :] = bottom_mask 
        combined_mask[:, 0:edge_thickness] = left_mask
        combined_mask[:, screen_width-edge_thickness:] = right_mask

        if (detect() == 100):
            hp = 100

        cv2.imshow("Red Detection at Screen Edges", combined_mask)

        if ((top_red_pixels > 1000) and (bottom_red_pixels > 1000) and (left_red_pixels > 1000) and (right_red_pixels > 1000)) :
            print("DEAD")

        if top_red_pixels > 1000:
            # print("Red patch detected at the top!")
            if (detect() < hp): 
                hp = detect()
                return True
                
        if bottom_red_pixels > 1000:
            # print("Red patch detected at the bottom!")
            if (detect() < hp): 
                hp = detect()
                return True

        if left_red_pixels > 1000:
            # print("Red patch detected on the left!")
            if (detect() < hp): 
                hp = detect()
                return True

        if right_red_pixels > 1000:
            # print("Red patch detected on the right!")
            if (detect() < hp): 
                hp = detect()
                return True
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()