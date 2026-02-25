import numpy as np
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

size = 100
image = np.zeros((size, size, 3), dtype="uint8")
assert image.shape[0] == image.shape[1]

color1 = np.array([255, 128, 0])
color2 = np.array([0, 128, 255])

max_dist = (size - 1) + (size - 1)

for y in range(size):
    for x in range(size):
        t = (x + y) / max_dist
        color = (lerp(color2, color1, t))
        image[y, x] = color

plt.figure(1)
plt.imshow(image)
plt.show()