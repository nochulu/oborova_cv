import cv2
import numpy as np
from pathlib import Path
import json


save_path = Path(__file__).parent
config_path = save_path / "main.json"

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

target = [1, 0, 2]
colors = {}
colors_name = {0:"blue", 1:"red",2: "green",3: "yellow"}
positions = []
while cam.isOpened():
    ret,frame=cam.read()
    blur = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    if clicked:
        clicked = False
        color = hsv[position[1]][position[0]]
        lower = np.clip(color * 0.9, 0, 255).astype("u1")
        upper = np.clip(color * 1.1, 0, 255).astype("u1")
        upper[1] = 255
        upper[2] = 255

    if key == ord('b'):
        colors[0] = (lower.copy(), upper.copy())

    if key == ord('r'):
        colors[1] = (lower.copy(), upper.copy())

    if key == ord('g'):
        colors[2] = (lower.copy(), upper.copy())

    if key == ord('y'):
        colors[3] = (lower.copy(), upper.copy())

    founded_balls = []
    for colors_id, (l, u) in colors.items():
        inr = cv2.inRange(hsv, l, u)
        mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("Mask", mask)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x,y), radius = cv2.minEnclosingCircle(contour)
            if cv2.contourArea(contour) > 500:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                x, y = int(x), int(y)
                cv2.circle(frame, (x, y), int(radius), (255, 255, 255), 2)
                name = colors_name.get(colors_id, str(colors_id))
                cv2.putText(frame, name, (x, y - 10), 0, 0.7, (255, 255, 255), 2)

                founded_balls.append((x, colors_id))

    founded_balls.sort()
    current_order = [ball[1] for ball in founded_balls]
    if current_order == target:
        cv2.putText(frame, "OTGADAL!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
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
