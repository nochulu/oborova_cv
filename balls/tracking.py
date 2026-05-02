import cv2
import numpy as np
from math import dist
import time
from pathlib import Path
import json

save_path = Path(__file__).parent
config_path = save_path / "config.json"

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

position = (0,0)
clicked = False
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")
        global position
        global clicked
        position = (x, y)
        clicked = True

cv2.setMouseCallback("Image", on_click)
cam = cv2.VideoCapture(0)
lower = None
upper = None

if config_path.exists():
    with config_path.open("r") as f:
        js = json.load(f)
        if "lower" in js:
            lower = np.array(js["lower"], dtype="u1")
            upper = np.array(js["upper"], dtype="u1")
positions = []
prev_time = time.time()
curr_time = time.time()
d = 6.36 #cm
while cam.isOpened():
    ret,frame=cam.read()
    blur = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    if clicked:
        clicked = False
        color = hsv[position [1]][position [0]]
        lower = np.clip(color * 0.9, 0, 255).astype("u1")
        upper = np.clip(color * 1.1, 0, 255).astype("u1")

    if lower is not None:
        inr = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5)))
        cv2.imshow("Mask", mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x,y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                x = int(x)
                y = int(y)
                radius = int(radius)
                cv2.circle(frame, (x, y), radius, (128, 0, 128), 2)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                positions.append((x,y))
                if len(positions) > 20:
                    positions.pop(0)
                for i,pos in enumerate(positions [:-1]):
                    cv2.circle(frame, pos, i*2, (0, 0, 100  + 155 / len(positions) * i ), -1)

                curr_time = time.time()
                delta_time = curr_time - prev_time
                if len(positions) >= 2:
                    curr_position = positions[-1]
                    prev_position = positions[-2]
                    dst = dist(curr_position, prev_position)
                    pxl_per_cm = d / (2 * radius)
                    pxl_per_m = pxl_per_cm / 100
                    speed = (dst / delta_time) * pxl_per_m
                    cv2.putText(frame, f"Speed: {speed:.2f} m/s", (10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2 )
                prev_time = curr_time
    cv2.imshow("Image", frame)
cam.release()
cv2.destroyAllWindows()

with config_path.open("w") as f:
    json.dump(
        {"lower":None if lower is None else lower.tolist(),
         "upper":None if upper is None else upper.tolist(),
        },
        f
    )