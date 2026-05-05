import cv2
import numpy as np
import os
import glob
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

#path
DATASET_PATH = r"C:\Users\Lenovo\Desktop\Image Processing\Project\Gravity Falls"
ARTIFACTS_PATH = os.path.join(DATASET_PATH, "processed_artifacts")
VISUALS_PATH = os.path.join(DATASET_PATH, "report_visuals")

class PuzzleProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.filename = os.path.basename(image_path)
        
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Could not load: {image_path}")

    
        # fixing lighting with lab colour space
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_new = clahe.apply(l)
        
        merged = cv2.merge((l_new, a, b))
        self.enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        #using bilateral filter to clean up the noise by smoothing textures
        self.smoothed = cv2.bilateralFilter(self.enhanced, 9, 75, 75)
        
        #sharpening for edge detection
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.final_img = cv2.filter2D(self.smoothed, -1, kernel)

        #binaryedgemap
        self.edges = cv2.Canny(self.final_img, 100, 200)

        # calculating gradients to find the straight lines for the grid cuts
        self.h, self.w = self.final_img.shape[:2]
        lab_float = cv2.cvtColor(self.final_img, cv2.COLOR_BGR2LAB).astype("float32")
        self.grad_x = np.sum(np.abs(np.diff(lab_float, axis=1)), axis=2)
        self.grad_y = np.sum(np.abs(np.diff(lab_float, axis=0)), axis=2)

    def check_grid_alignment(self, n_cuts):
        #summing up the gradients along rows/cols to find cuts
        col_prof = np.sum(self.grad_x, axis=0)
        row_prof = np.sum(self.grad_y, axis=1)
        
        # Normalizing the profiles
        col_prof = col_prof / (np.median(col_prof) + 1e-5)
        row_prof = row_prof / (np.median(row_prof) + 1e-5)
        
        total_score = 0
        checks = 0
        window = 2
        
        step_w = self.w / n_cuts
        step_h = self.h / n_cuts
        
        # We only care about the inner cuts, not the borders of the image
        intervals = [i for i in range(1, n_cuts) if i % 2 != 0]
        
        for i in intervals:
            # check if there's a vertical line here
            x = int(i * step_w)
            start_x, end_x = max(0, x - window), min(len(col_prof), x + window + 1)
            total_score += np.max(col_prof[start_x:end_x])
            checks += 1
            
            # Check if there's a horizontal line here
            y = int(i * step_h)
            start_y, end_y = max(0, y - window), min(len(row_prof), y + window + 1)
            total_score += np.max(row_prof[start_y:end_y])
            checks += 1
            
        #returning average score for this grid size
        return total_score / checks if checks > 0 else 0

    def guess_size(self):
        # check if the image lines up better with an 8x8 or 4x4 or 2x2grid.
        score_8 = self.check_grid_alignment(8)
        score_4 = self.check_grid_alignment(4)
        
        if score_8 > 2.0: return 8
        elif score_4 > 2.5: return 4
        else: return 2

    def export_tiles(self, n):
        folder_name = f"detected_{n}x{n}"
        name = os.path.splitext(self.filename)[0]
        out_path = os.path.join(ARTIFACTS_PATH, folder_name, name)
        
        if os.path.exists(out_path): shutil.rmtree(out_path)
        os.makedirs(out_path)

        h_step = self.h // n
        w_step = self.w // n
        
        for i in range(n):
            for j in range(n):
                tile = self.img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                cv2.imwrite(os.path.join(out_path, f"tile_{i}_{j}.png"), tile)
        
        cv2.imwrite(os.path.join(out_path, "contours.png"), self.edges)

    def create_debug_plot(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(self.final_img, cv2.COLOR_BGR2RGB))
        plt.title("Enhanced")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(self.edges, cmap='gray')
        plt.title("Contours")
        plt.axis('off')
        
        os.makedirs(VISUALS_PATH, exist_ok=True)
        plt.savefig(os.path.join(VISUALS_PATH, f"pipeline_{self.filename}.png"))
        plt.close()

def process_dataset():
    print("Starting Processing...")
   
    if os.path.exists(ARTIFACTS_PATH): shutil.rmtree(ARTIFACTS_PATH)
    if os.path.exists(VISUALS_PATH): shutil.rmtree(VISUALS_PATH)
    
    folders = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']
    
    for folder in folders:
        path = os.path.join(DATASET_PATH, folder)
        if not os.path.exists(path): continue
        
    
        files = [f for f in glob.glob(os.path.join(path, "*.*")) 
                 if f.lower().endswith(('.jpg','.png','.jpeg'))]
        
        true_val = int(folder.split('_')[1][0])
        correct_count = 0
        plots_made = 0
        
        print(f"\nFolder: {folder}")
        
        for f in tqdm(files):
            try:
                proc = PuzzleProcessor(f)
                result = proc.guess_size()
                
                proc.export_tiles(result)
                
                if result == true_val:
                    correct_count += 1
                
                if plots_made < 2:
                    proc.create_debug_plot()
                    plots_made += 1
                    
            except Exception as e:
                print(f"Error on {f}: {e}")
        
        acc = (correct_count / len(files) * 100) if files else 0
        print(f"Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    process_dataset()