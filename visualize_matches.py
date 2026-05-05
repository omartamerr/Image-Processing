import cv2
import numpy as np
import os
import glob
import re
import heapq
import math
import random

DATASET_PATH = r"C:\Users\Lenovo\Desktop\Image Processing\Project\Gravity Falls"
ARTIFACTS_PATH = os.path.join(DATASET_PATH, "processed_artifacts")
OUTPUT_IMAGE = os.path.join(DATASET_PATH, "match_visualization.png")

GRID_TO_VISUALIZE = 4 

class PuzzleAssembler:
    def __init__(self, folder, n):
        self.n = n
        self.tiles = []
        files = glob.glob(os.path.join(folder, "*.png"))
        
        files = [f for f in files if "tile_" in os.path.basename(f)]
        
        def natural_key(s): return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]
        files.sort(key=natural_key)

        for f in files:
            img = cv2.imread(f)
            if img is not None:
                h, w = img.shape[:2]
                if h > 4 and w > 4: 
                    crop = 1
                    img = img[crop:h-crop, crop:w-crop]
                
                self.tiles.append({
                    'id': len(self.tiles),
                    'img': img, 
                    'lab': cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
                })

    def calc_pair_cost_v14(self, t1, t2, direction):
        if direction == 0: 
            b1 = t1['lab'][:, -1, :]; b2 = t2['lab'][:, 0, :]
            d1 = t1['lab'][:, -2, :]; d2 = t2['lab'][:, 1, :]
        else: 
            b1 = t1['lab'][-1, :, :]; b2 = t2['lab'][0, :, :]
            d1 = t1['lab'][-2, :, :]; d2 = t2['lab'][1, :, :]

        diff = b1 - b2
        cost_p = np.sum(np.sqrt(np.sum(diff**2, axis=1)))
        
        grad_t1 = b1 - d1
        grad_t2 = d2 - b2
        grad_seam = b2 - b1
        cost_g = np.sum(np.sqrt(np.sum((grad_seam - grad_t1)**2, axis=1))) + \
                 np.sum(np.sqrt(np.sum((grad_seam - grad_t2)**2, axis=1)))
        
        return cost_p + 1.5 * cost_g

def find_puzzle_folder(artifacts_path, n):
    base = os.path.join(artifacts_path, f"detected_{n}x{n}")
    if not os.path.exists(base):
        print(f"[ERROR] Could not find folder: {base}")
        print("Make sure you ran MS1 first!")
        return None

    for p in sorted(os.listdir(base)):
        puzzle_dir = os.path.join(base, p)
        if os.path.isdir(puzzle_dir):
            pngs = glob.glob(os.path.join(puzzle_dir, "*.png"))
            tiles_only = [f for f in pngs if "tile_" in f]
            if len(tiles_only) == n * n:
                return puzzle_dir
    return None

def visualize_best_buddies():
    print(f"--- Visualizing {GRID_TO_VISUALIZE}x{GRID_TO_VISUALIZE} Matches ---")
    
    puzzle_dir = find_puzzle_folder(ARTIFACTS_PATH, GRID_TO_VISUALIZE)
    if not puzzle_dir:
        print("No valid puzzle folder found. Exiting.")
        return

    print(f"[INFO] Analyzing puzzle: {puzzle_dir}")

    asm = PuzzleAssembler(puzzle_dir, GRID_TO_VISUALIZE)
    tiles = asm.tiles
    nt = len(tiles)
    print(f"[INFO] Loaded {nt} tiles.")

    if nt == 0:
        print("[ERROR] No tiles found inside the folder.")
        return

    print("[INFO] Computing costs...")
    costs_h = np.full((nt, nt), np.inf)
    costs_v = np.full((nt, nt), np.inf)

    for i in range(nt):
        for j in range(nt):
            if i != j:
                costs_h[i, j] = asm.calc_pair_cost_v14(tiles[i], tiles[j], 0)
                costs_v[i, j] = asm.calc_pair_cost_v14(tiles[i], tiles[j], 1)

    buddies = []
    
    best_h_fwd = np.argmin(costs_h, axis=1)
    best_h_bwd = np.argmin(costs_h, axis=0)
    for i in range(nt):
        j = best_h_fwd[i]
        if best_h_bwd[j] == i:
            buddies.append((i, j, "H"))

    best_v_fwd = np.argmin(costs_v, axis=1)
    best_v_bwd = np.argmin(costs_v, axis=0)
    for i in range(nt):
        j = best_v_fwd[i]
        if best_v_bwd[j] == i:
            buddies.append((i, j, "V"))

    print(f"[INFO] Found {len(buddies)} 'Best Buddy' connections.")

    h, w, _ = tiles[0]['img'].shape
    margin = 20
    cols = GRID_TO_VISUALIZE
    rows = int(np.ceil(nt / cols))

    canvas_h = rows * (h + margin) + margin
    canvas_w = cols * (w + margin) + margin
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    centers = {}

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y = margin + r * (h + margin)
        x = margin + c * (w + margin)
        
        canvas[y:y+h, x:x+w] = tile['img']
        centers[idx] = (x + w // 2, y + h // 2)

        cv2.putText(
            canvas, str(idx),
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2
        )

    for i, j, d in buddies:
        p1 = centers[i]
        p2 = centers[j]
        color = (0, 255, 0) if d == "H" else (255, 0, 0)
        cv2.line(canvas, p1, p2, color, 2)
        cv2.circle(canvas, p1, 4, color, -1)
        cv2.circle(canvas, p2, 4, color, -1)

    cv2.imwrite(OUTPUT_IMAGE, canvas)
    print(f"[SUCCESS] Saved visualization to: {os.path.abspath(OUTPUT_IMAGE)}")

if __name__ == "__main__":
    visualize_best_buddies()