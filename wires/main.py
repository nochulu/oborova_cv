import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import opening

image = np.load("C:/Users/oboro/OneDrive/Desktop/isu/python/wires/wires3.npy")
struct = np.ones((3,1))
process = opening(image, struct)


labeled_image = label(image)
labeled_process = label(process)
print(f"Original{np.max(labeled_image)}")
print(f"Processed{np.max(labeled_process)}")

for wires_id in range (1, np.max(labeled_image) + 1 ):
    wire = labeled_image == wires_id
    parts = opening(wire, struct)
    labeled_parts = label(parts)
    parts_count = np.max(labeled_parts)
    print(f"Wire #6+{wires_id} {parts_count} parts")

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(process)
plt.show()