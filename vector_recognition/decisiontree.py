import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.distance_measures import eccentricity
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path


def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1


def vertical_symmetry(region):
    img = region.image
    w = img.shape[1]
    mid = w // 2
    left_part = img[:, :mid]
    right_part = img[:, w - mid:]
    left_sum = np.sum(left_part)
    right_sum = np.sum(right_part)
    if left_sum == 0: return 0
    return left_sum / right_sum


def classificator(region):
    holes = count_holes(region)
    ratio = vertical_symmetry(region)
    if holes == 2:  # B, 8
        #vlines = (np.sum(region.image, 0) == region.image.shape[0]).sum()
        #vlines = vlines / region.image.shape[1]
        if ratio > 1.1:
            return "B"
        else:
            return "8"
    elif holes == 1:  # A, O
        if region.eccentricity > 0.58:
            return "O"
        else:
            return "A"
    else:  # 1, W, X, *, -, /
        if region.image.sum() / region.image.size == 1:
            return "-"
        shape = region.image.shape
        aspect = np.min(shape) / np.max(shape)

        if aspect > 0.9:
            return "*"

        vlines = (np.sum(region.image, 0) == region.image.shape[0]).sum()
        hlines = (np.sum(region.image, 1) == region.image.shape[1]).sum()
        if vlines > 0 and hlines > 0:
            return "1"

        labeled = label(np.logical_not(region.image))
        bays = 0
        for r in regionprops(labeled):
            if r.area > 3:
                bays += 1
        if bays == 5:
            return "W"
        elif bays == 4:
            return "X"
        elif bays == 2:
            return "/"
    return "?"


save_path = Path(__file__).parent

image = imread("alphabet.png")[:, :, :-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
print(f"Total objects: {np.max(alabeled)}")
aprops = regionprops(alabeled)

results = {}
image_path = save_path / "out_tree"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5, 7))
for region in aprops:
    symbol = classificator(region)
    if symbol not in results:
        results[symbol] = 0
    results[symbol] += 1

    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")

print(f"Results: {results}")
print(f"Accuracy: {1.0 - results.get('?', 0) / len(aprops)}")