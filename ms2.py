import cv2
import numpy as np
import os
import glob
import shutil
import itertools
import re
import heapq
import math
import random
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\Lenovo\Desktop\Image Processing\Project\Gravity Falls"

ARTIFACTS_PATH = os.path.join(DATASET_PATH, "processed_artifacts")
SOLVED_PATH = os.path.join(DATASET_PATH, "final_assembly_v16_sa_hybrid")
GROUND_TRUTH_PATH = os.path.join(DATASET_PATH, "correct")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"Processing: {desc}")
        return iterable

class PuzzleAssembler:
    def __init__(self, folder, n):
        self.n = n; self.tiles = []
        files = glob.glob(os.path.join(folder, "*.png"))
    
        files = [f for f in files if "tile_" in os.path.basename(f)]
        # ----------------------------------------------------------

        def natural_key(s): return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]
        files.sort(key=natural_key)

        for f in files:
            img = cv2.imread(f)
            if img is not None:
                self.tiles.append({
                    'id': len(self.tiles),
                    'img': img, 
                    'lab': cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
                })
        
        if self.n == 2:
            for t in self.tiles:
                t['sa_feats'] = self._compute_all_sa_edges(t['img'])

    def _compute_sa_edge_feature(self, img_bgr, side, strip_width=6):
        h, w = img_bgr.shape[:2]
        strip_width = max(2, min(strip_width, h // 4, w // 4))
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab)
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        G = cv2.magnitude(gx, gy)
        
        TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
        if side == TOP:
            strips = [ch[0:strip_width, :].mean(axis=0) for ch in (L, A, B, G)]
        elif side == BOTTOM:
            strips = [ch[h-strip_width:h, :].mean(axis=0) for ch in (L, A, B, G)]
        elif side == LEFT:
            strips = [ch[:, 0:strip_width].mean(axis=1) for ch in (L, A, B, G)]
        else:
            strips = [ch[:, w-strip_width:w].mean(axis=1) for ch in (L, A, B, G)]
            
        normalized = []
        for s in strips:
            s = s.astype(np.float32)
            std = s.std()
            normalized.append((s - s.mean()) / (std + 1e-6) if std > 1e-6 else s * 0.0)
        return np.concatenate(normalized).astype(np.float32)

    def _compute_all_sa_edges(self, img):
        return {side: self._compute_sa_edge_feature(img, side) for side in range(4)}

    def _sa_edge_cost(self, f1, f2):
        diff = f1 - f2
        return float(np.dot(diff, diff))

    def solve_simulated_annealing(self):
        TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
        costs = {}
        for t1 in self.tiles:
            for t2 in self.tiles:
                if t1['id'] == t2['id']: continue
                costs[(t1['id'], RIGHT, t2['id'], LEFT)] = self._sa_edge_cost(t1['sa_feats'][RIGHT], t2['sa_feats'][LEFT])
                costs[(t1['id'], LEFT, t2['id'], RIGHT)] = self._sa_edge_cost(t1['sa_feats'][LEFT], t2['sa_feats'][RIGHT])
                costs[(t1['id'], BOTTOM, t2['id'], TOP)] = self._sa_edge_cost(t1['sa_feats'][BOTTOM], t2['sa_feats'][TOP])
                costs[(t1['id'], TOP, t2['id'], BOTTOM)] = self._sa_edge_cost(t1['sa_feats'][TOP], t2['sa_feats'][BOTTOM])

        def assignment_cost(perm, N):
            total = 0.0
            for r in range(N):
                for c in range(N):
                    tid = perm[r * N + c]
                    if c + 1 < N:
                        total += costs[(tid, RIGHT, perm[r * N + c + 1], LEFT)]
                    if r + 1 < N:
                        total += costs[(tid, BOTTOM, perm[(r + 1) * N + c], TOP)]
            return total

        tile_ids = [t['id'] for t in self.tiles]
        N = self.n
        best_perm, best_cost = None, float("inf")
        
        restarts = 25
        iters = 2000
        random.seed(0)

        for r in range(restarts):
            perm = tile_ids[:]
            random.shuffle(perm)
            cur_cost = assignment_cost(perm, N)
            restart_best_perm, restart_best_cost = perm[:], cur_cost

            T0, T_end = 5.0, 0.1

            for it in range(iters):
                i, j = random.sample(range(N * N), 2)
                new_perm = perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_cost = assignment_cost(new_perm, N)
                
                T = T0 * (T_end / T0) ** (it / max(1, iters - 1))
                
                if new_cost < cur_cost or random.random() < math.exp(-(new_cost - cur_cost) / max(T, 1e-6)):
                    perm, cur_cost = new_perm, new_cost
                    if cur_cost < restart_best_cost:
                        restart_best_perm, restart_best_cost = perm[:], cur_cost

            if restart_best_cost < best_cost:
                best_perm, best_cost = restart_best_perm, restart_best_cost
        
        best_grid = np.zeros((N, N), dtype=int)
        for r in range(N):
            for c in range(N):
                best_grid[r, c] = best_perm[r * N + c]
        return best_grid

    def calc_pair_cost_v14(self, t1, t2, direction):
        if direction == 0:
            b1 = t1['lab'][:, -1, :]; b2 = t2['lab'][:, 0, :]
            d1 = t1['lab'][:, -2, :]; d2 = t2['lab'][:, 1, :]
        else:
            b1 = t1['lab'][-1, :, :]; b2 = t2['lab'][0, :, :]
            d1 = t1['lab'][-2, :, :]; d2 = t2['lab'][1, :, :]

        diff = b1 - b2
        cost_p = np.sum(np.sqrt(np.sum(diff**2, axis=1)))
        grad_t1 = b1 - d1; grad_t2 = d2 - b2; grad_seam = b2 - b1
        cost_g = np.sum(np.sqrt(np.sum((grad_seam - grad_t1)**2, axis=1))) + \
                 np.sum(np.sqrt(np.sum((grad_seam - grad_t2)**2, axis=1)))
        
        return cost_p + 1.5 * cost_g

    def solve_v14_hybrid(self):
        nt = len(self.tiles)
        costs = np.zeros((nt, nt, 2))
        for i in range(nt):
            for j in range(nt):
                if i != j:
                    costs[i][j][0] = self.calc_pair_cost_v14(self.tiles[i], self.tiles[j], 0)
                    costs[i][j][1] = self.calc_pair_cost_v14(self.tiles[i], self.tiles[j], 1)
        
        buddies_h = set()
        buddies_v = set()
        best_h_fwd = np.argmin(costs[:,:,0], axis=1)
        best_h_bwd = np.argmin(costs[:,:,0], axis=0)
        for i in range(nt):
            j = best_h_fwd[i]
            if best_h_bwd[j] == i: buddies_h.add((i, j))

        best_v_fwd = np.argmin(costs[:,:,1], axis=1)
        best_v_bwd = np.argmin(costs[:,:,1], axis=0)
        for i in range(nt):
            j = best_v_fwd[i]
            if best_v_bwd[j] == i: buddies_v.add((i, j))

        global_best_grid = None
        global_max_confidence = -1.0 

        seeds = range(nt)
        for seed in seeds:
            grid = np.full((self.n, self.n), -1, dtype=int)
            grid[0, 0] = seed
            used = {seed}
            current_grid_confidence = 0.0
            pq = []
            
            def add_neighbors(r, c):
                if c + 1 < self.n and grid[r, c+1] == -1: push_cand(r, c+1)
                if r + 1 < self.n and grid[r+1, c] == -1: push_cand(r+1, c)
                if c - 1 >= 0 and grid[r, c-1] == -1: push_cand(r, c-1)
                if r - 1 >= 0 and grid[r-1, c] == -1: push_cand(r-1, c)

            def push_cand(r, c):
                tL = grid[r, c-1] if c > 0 else -1
                tR = grid[r, c+1] if c < self.n-1 else -1
                tT = grid[r-1, c] if r > 0 else -1
                tB = grid[r+1, c] if r < self.n-1 else -1
                
                candidates = []
                for cand in range(nt):
                    if cand in used: continue
                    cost = 0
                    is_buddy = False

                    if tL != -1: 
                        cost += costs[tL][cand][0]
                        if (tL, cand) in buddies_h: is_buddy = True
                    if tR != -1: 
                        cost += costs[cand][tR][0]
                        if (cand, tR) in buddies_h: is_buddy = True
                    if tT != -1: 
                        cost += costs[tT][cand][1]
                        if (tT, cand) in buddies_v: is_buddy = True
                    if tB != -1: 
                        cost += costs[cand][tB][1]
                        if (cand, tB) in buddies_v: is_buddy = True
                    
                    if is_buddy: cost *= 0.1

                    candidates.append((cost, cand))
                
                if not candidates: return
                candidates.sort(key=lambda x: x[0])
                
                best_cost = candidates[0][0]
                best_cand = candidates[0][1]
                
                ratio = 1.0
                if len(candidates) > 1 and candidates[1][0] > 1e-5:
                    ratio = best_cost / candidates[1][0]
                
                heapq.heappush(pq, (ratio, best_cost, r, c, best_cand))

            add_neighbors(0, 0)
            
            while len(used) < nt and pq:
                ratio, cost, r, c, cand = heapq.heappop(pq)
                if grid[r, c] != -1: continue
                if cand in used: 
                    push_cand(r, c)
                    continue

                grid[r, c] = cand
                used.add(cand)
                current_grid_confidence += (1.0 - ratio)
                add_neighbors(r, c)

            if len(used) == nt:
                if current_grid_confidence > global_max_confidence:
                    global_max_confidence = current_grid_confidence
                    global_best_grid = grid.copy()

        return global_best_grid

    def solve(self):
        if self.n == 2: 
            return self.solve_simulated_annealing()
        else: 
            return self.solve_v14_hybrid()

    def reconstruct(self, grid):
        if grid is None: return None
        rows = []
        for r in range(self.n):
            row_imgs = []
            for c in range(self.n):
                idx = grid[r, c]
                row_imgs.append(self.tiles[idx]['img'])
            rows.append(np.hstack(row_imgs))
        return np.vstack(rows)

def get_id(fname):
    nums = re.findall(r'\d+', fname)
    return int(nums[-1]) if nums else -1

def run_project():
    print("=== FINAL PROJECT ===")

    print("\n[MS2] Assembly...")
    if os.path.exists(SOLVED_PATH): shutil.rmtree(SOLVED_PATH)
    
    det_folders = glob.glob(os.path.join(ARTIFACTS_PATH, "detected_*"))
    for df in det_folders:
        try:
            n = int(os.path.basename(df).split('_')[1][0])
            out = os.path.join(SOLVED_PATH, os.path.basename(df))
            os.makedirs(out, exist_ok=True)
            for p_dir in tqdm(glob.glob(os.path.join(df, "*")), desc=f"Grid {n}x{n}"):
                if not os.path.isdir(p_dir): continue 
                asm = PuzzleAssembler(p_dir, n)
                res = asm.reconstruct(asm.solve())
                if res is not None:
                    cv2.imwrite(os.path.join(out, os.path.basename(p_dir) + ".png"), res)
        except Exception as e:
            print(f"Error: {e}")

    print("\n[EVAL] Calculating Accuracy...")
    stats = {}
    folders = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']
    
    for folder in folders:
        in_dir = os.path.join(DATASET_PATH, folder)
        if not os.path.exists(in_dir): continue
        
        exp_n = folder.split('_')[1]
        sol_dir = os.path.join(SOLVED_PATH, f"detected_{exp_n}")
        
        total = len(glob.glob(os.path.join(in_dir, "*.*")))
        correct = 0
        
        if os.path.exists(sol_dir):
            for gt in glob.glob(os.path.join(GROUND_TRUTH_PATH, "*.*")):
                gt_id = get_id(os.path.basename(gt))
                match = None
                for s in glob.glob(os.path.join(sol_dir, "*.png")):
                    if get_id(os.path.basename(s)) == gt_id:
                        match = s; break
                
                if match:
                    img_s = cv2.imread(match)
                    img_g = cv2.imread(gt)
                    if img_s is not None and img_g is not None:
                        if img_s.shape != img_g.shape:
                            img_g = cv2.resize(img_g, (img_s.shape[1], img_s.shape[0]))
                        err = np.mean((img_s.astype("float") - img_g.astype("float")) ** 2)
                        if err < 3500: correct += 1
        
        acc = (correct / total * 100) if total > 0 else 0
        stats[folder] = acc
        print(f"  {folder}: {correct}/{total} ({acc:.2f}%)")

    plt.figure(figsize=(8,5))
    plt.bar(stats.keys(), stats.values(), color=['#E91E63','#9C27B0','#673AB7'])
    plt.ylim(0,105); plt.ylabel("Accuracy %"); plt.title("Final Test")
    for i,v in enumerate(stats.values()): plt.text(i,v+1,f"{v:.1f}%",ha='center')
    plt.savefig(os.path.join(DATASET_PATH, "final_score.png"))
    plt.show()

if __name__ == "__main__":
    run_project()