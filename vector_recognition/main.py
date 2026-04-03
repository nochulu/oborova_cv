import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.distance_measures import eccentricity
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

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


def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2,shape[1] + 2))
    new_image [1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def extractor(region):
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter/region.image.size
    holes = count_holes(region)
    vlines = (np.sum(region.image,0) == region.image.shape[0]).sum() / region.image.shape[1]
    hlines = (np.sum(region.image,1) == region.image.shape[1]).sum() / region.image.shape[0]
    eccentricity = region.eccentricity
    h, w = region.image.shape
    aspect = min(h, w) / max(h, w)
    v_sym = vertical_symmetry(region)
    return np.array([region.area/region.image.size, cx, cy, perimeter, holes, vlines, hlines, eccentricity, aspect, v_sym])

def classificator(region, templates):
    features = extractor(region)
    result = ""
    min_d = 10 ** 16
    for symbol, t in templates.items():
        d = ((t - features)**2).sum() ** 0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result

template = imread("alphabet-small.png")[:,:,:-1]
#print(template.shape)
template = template.sum(2)
binary = template != 765.

labeled = label(binary)
props = regionprops(labeled)
#print(type(props))

templates = {}

symbols = ["A", "B", "8", "0", "1", "W", "X", "*", "-", "/"]
for region, symbol in zip(props, symbols):
    templates[symbol] = extractor(region)

image = imread("alphabet.png")[:,:,:-1]
abinary = image.mean(2)>0
alabeled = label(abinary)
print(np.max(alabeled))
aprops = regionprops(alabeled)
results = {}
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)
#plt.ion()
plt.figure(figsize=(5,7))
for region in aprops:
    symbol = classificator(region, templates)
    if symbol not in results:
        results[symbol] = 0
    results[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")
print(results)

#print(templates)
print(classificator(props[0],templates))
#print(count_holes(aprops[0]))
#plt.imshow(abinary)
#plt.show()