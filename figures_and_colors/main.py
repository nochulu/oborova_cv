import cv2
import numpy as np

image = cv2.imread("balls_and_rects.png")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def classify(contour):
    perimetr = cv2.arcLength(contour, True)
    if perimetr == 0:
        return "None"

    _, radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    circle_area = np.pi * radius ** 2
    solidity = area / circle_area

    if solidity > 0.8:
        return "circle"

    eps = 0.04 * perimetr
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) == 4:
        return "rectangle"
    return "None"

def get_hue_label(contour, hsv_img):
    mask = np.zeros(hsv_img.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(hsv_img, mask=mask)
    return int(mean_val[0])

circles_by_color = {}
rects_by_color = {}

for cnt in contours:
    if cv2.contourArea(cnt) < 10:
        continue

    figure = classify(cnt)
    if figure == "None":
        continue

    hue = get_hue_label(cnt, hsv)
    hue_group = (hue // 2) * 2

    if figure == "circle":
        circles_by_color[hue_group] = circles_by_color.get(hue_group, 0) + 1
    elif figure == "rectangle":
        rects_by_color[hue_group] = rects_by_color.get(hue_group, 0) + 1

total_circles = sum(circles_by_color.values())
total_rectangles = sum(rects_by_color.values())

print(f"Всего фигур: {total_circles + total_rectangles}")
print(f"Всего кругов: {total_circles}")
print(f"Всего прямоугольников: {total_rectangles}")

for h, count in sorted(circles_by_color.items()):
    print(f"Круг (оттенок ~{h}): {count}")

for h, count in sorted(rects_by_color.items()):
    print(f"Прямоугольник (оттенок ~{h}): {count}")