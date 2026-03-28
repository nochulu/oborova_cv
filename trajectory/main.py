import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from pathlib import Path

folder = Path("motion/out")
traectories = {}
next_id = 0

for i in range(100):
    file_path = folder / f"h_{i}.npy"
    if not file_path.exists():
        continue

    points = np.load(file_path)
    props = regionprops(label(points))
    centroids = [p.centroid for p in props]
    used_now = set()

    for t_id, path in traectories.items():
        if not centroids:
            break

        last_pos = path[-1]
        dists = []
        for c in centroids:
            dx = last_pos[0] - c[0]
            dy = last_pos[1] - c[1]
            d = (dx ** 2 + dy ** 2) ** 0.5
            dists.append(d)

        min_dist = dists[0]
        best_idx = 0
        for j in range(1, len(dists)):
            if dists[j] < min_dist:
                min_dist = dists[j]
                best_idx = j

        if min_dist < 20:
            if best_idx not in used_now:
                traectories[t_id].append(centroids[best_idx])
                used_now.add(best_idx)

    for idx in range(len(centroids)):
        if idx not in used_now:
            new_point = centroids[idx]
            traectories[next_id] = [new_point]
            next_id += 1

plt.figure(figsize=(10, 10))

for path in traectories.values():
    if len(path) > 1:
        pts = np.array(path)
        plt.plot(pts[:, 1], pts[:, 0], "-")

plt.title(f"Всего найдено объектов: {next_id}")
plt.show()

