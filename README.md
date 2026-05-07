# Image Processing - Puzzle Detection and Assembly

A sophisticated image processing project that automatically detects puzzle grids, extracts puzzle tiles, and intelligently reconstructs complete puzzles using advanced computer vision techniques.

## 📋 Overview

This project implements a two-stage pipeline for puzzle solving:
- **Stage 1 (MS1)**: Detects puzzle grid size and extracts individual tiles
- **Stage 2 (MS2)**: Assembles tiles back into complete puzzles using optimization algorithms

The system is designed to handle puzzle images of varying complexities, specifically tested on 2×2, 4×4, and 8×8 grid puzzles.

## 🎯 Features

### Image Enhancement
- **Lighting Correction**: Uses LAB color space with CLAHE (Contrast Limited Adaptive Histogram Equalization) to fix uneven lighting
- **Noise Reduction**: Bilateral filtering to smooth textures while preserving edges
- **Sharpening**: Kernel-based sharpening for enhanced edge detection

### Grid Detection
- **Gradient Analysis**: Computes gradients in LAB color space to identify grid lines
- **Automatic Grid Sizing**: Intelligently determines whether puzzle is 2×2, 4×4, or 8×8
- **Edge Detection**: Canny edge detection for precise grid boundary identification

### Tile Extraction
- Automatically segments images into individual tiles
- Exports each tile as a separate PNG file
- Preserves original tile quality and positioning information

### Puzzle Assembly
**Two Specialized Solving Algorithms**:

1. **Simulated Annealing** (for 2×2 puzzles):
   - Heuristic optimization approach
   - Edge feature matching based on LAB color gradients
   - 25 restarts with 2000 iterations each
   - Computes normalized edge features for robust matching

2. **Hybrid Greedy-Search** (for 4×4 and 8×8 puzzles):
   - Buddy detection for mutual edge matches
   - Priority queue-based greedy placement
   - Confidence scoring for solution quality
   - Multiple seed positions for exploration

### Edge Matching Features
- **Simulated Annealing**: Uses Sobel gradients and LAB color information on edge strips
- **Hybrid Method**: Compares edge pixels with gradient-based cost calculation
- Both methods account for color continuity and gradient alignment at seams

## 📊 Project Structure

```
Image-Processing/
├── README.md              # This file
├── ms1.py                 # Stage 1: Grid detection and tile extraction
├── ms2.py                 # Stage 2: Tile assembly and puzzle reconstruction
├── visualize_matches.py   # Visualization utilities for debugging
├── demo.ipynb             # Jupyter notebook with demonstrations
├── Gravity Falls/          # Dataset directory
│   ├── puzzle_2x2/        # 2×2 puzzle samples
│   ├── puzzle_4x4/        # 4×4 puzzle samples
│   ├── puzzle_8x8/        # 8×8 puzzle samples
│   └── correct/           # Ground truth solutions
└── .gitignore            # Git ignore rules
```

## 🔧 Installation

### Requirements
- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib
- tqdm

### Setup
```bash
pip install opencv-python numpy matplotlib tqdm
```

## 🚀 Usage

### Stage 1: Grid Detection and Tile Extraction

```python
from ms1 import PuzzleProcessor, process_dataset

# Process all puzzles in the dataset
process_dataset()
```

The `process_dataset()` function will:
1. Scan puzzle folders (puzzle_2x2, puzzle_4x4, puzzle_8x8)
2. Detect the grid size for each image
3. Extract tiles and save them to `processed_artifacts/`
4. Generate debug visualizations in `report_visuals/`
5. Report accuracy of grid detection

### Stage 2: Puzzle Assembly

```python
from ms2 import run_project

# Run complete assembly pipeline
run_project()
```

The `run_project()` function will:
1. Load extracted tiles from Stage 1
2. Assemble puzzles using appropriate algorithm
3. Save reconstructed puzzles to `final_assembly_v16_sa_hybrid/`
4. Calculate accuracy metrics against ground truth
5. Generate performance visualization

### Individual Processing

```python
from ms1 import PuzzleProcessor

# Process a single image
proc = PuzzleProcessor("path/to/puzzle.jpg")
detected_size = proc.guess_size()  # Returns 2, 4, or 8
proc.export_tiles(detected_size)
proc.create_debug_plot()
```

## 📈 Algorithm Details

### Stage 1: Grid Detection

**Algorithm: Gradient Profile Analysis**

1. **Preprocessing**:
   - Load image and convert to LAB color space
   - Apply CLAHE for adaptive histogram equalization
   - Bilateral filter for noise reduction
   - Sharpen using unsharp mask kernel

2. **Gradient Computation**:
   - Calculate gradients along X and Y axes
   - Sum gradients across dimensions to create row/column profiles

3. **Grid Alignment Scoring**:
   - Test grid sizes: 8×8, 4×4, 2×2
   - For each size, check for strong gradients at expected cut positions
   - Calculate normalized scores at grid intersections

4. **Decision**:
   - 8×8 if score > 2.0
   - 4×4 if score > 2.5
   - Default to 2×2

### Stage 2: Puzzle Assembly

**For 2×2 Puzzles: Simulated Annealing**
- Initialize with edge feature costs between all tile pairs
- Use stochastic permutation search with temperature-based acceptance
- Temperature schedule: exponential cooling from 5.0 to 0.1
- Track best solution across multiple restarts

**For 4×4 and 8×8 Puzzles: Hybrid Greedy Search**
- Precompute edge matching costs using LAB color differences
- Identify "buddy pairs" (mutually best matches)
- Perform greedy placement with buddy-weighted costs
- Use multiple seed positions to escape local minima

## 📊 Performance Metrics

The system evaluates accuracy by:
- Comparing reconstructed puzzles with ground truth images
- Calculating Mean Squared Error (MSE) between images
- Threshold: MSE < 3500 for successful reconstruction
- Reporting accuracy as percentage of correctly assembled puzzles

## 🎨 Output Files

### From Stage 1:
- `processed_artifacts/detected_2x2/`: Extracted 2×2 puzzle tiles
- `processed_artifacts/detected_4x4/`: Extracted 4×4 puzzle tiles
- `processed_artifacts/detected_8x8/`: Extracted 8×8 puzzle tiles
- `report_visuals/`: Debug visualizations showing processing pipeline

### From Stage 2:
- `final_assembly_v16_sa_hybrid/`: Reconstructed complete puzzles
- `final_score.png`: Bar chart showing accuracy by puzzle size

## 🔍 Key Classes

### PuzzleProcessor (ms1.py)
Handles grid detection and tile extraction.

**Key Methods**:
- `__init__(image_path)`: Initialize and preprocess image
- `guess_size()`: Detect grid size (2, 4, or 8)
- `check_grid_alignment(n_cuts)`: Score grid alignment
- `export_tiles(n)`: Extract and save tiles
- `create_debug_plot()`: Generate visualization

### PuzzleAssembler (ms2.py)
Handles puzzle assembly and reconstruction.

**Key Methods**:
- `__init__(folder, n)`: Load tiles from folder
- `solve()`: Automatically select and run appropriate algorithm
- `solve_simulated_annealing()`: SA-based solver for 2×2
- `solve_v14_hybrid()`: Greedy hybrid solver for 4×4 and 8×8
- `reconstruct(grid)`: Assemble tiles into final image

## 🛠️ Configuration

Edit path variables in the scripts:

```python
# ms1.py and ms2.py
DATASET_PATH = r"path/to/your/dataset"
ARTIFACTS_PATH = os.path.join(DATASET_PATH, "processed_artifacts")
SOLVED_PATH = os.path.join(DATASET_PATH, "final_assembly_v16_sa_hybrid")
GROUND_TRUTH_PATH = os.path.join(DATASET_PATH, "correct")
```

## 📝 Jupyter Notebook

The `demo.ipynb` file contains:
- Step-by-step demonstrations of both stages
- Visualization of intermediate results
- Performance comparisons
- Code examples for custom usage

## 🎓 Technical Insights

### Why Two Algorithms?
- **Simulated Annealing** for small puzzles (2×2): Exhaustive search feasible
- **Greedy Hybrid** for larger puzzles (4×4, 8×8): Computational efficiency required

### Edge Matching Strategy
- Uses LAB color space for perceptually uniform color comparison
- Incorporates gradient information to detect seamless matches
- Normalizes features to handle lighting variations

### Optimization Techniques
- Buddy detection reduces search space
- Priority queue ensures best candidates are explored first
- Confidence scoring guides multi-seed exploration
- Adaptive feature extraction for different tile sizes

## 🐛 Troubleshooting

**Grid Detection Accuracy Low**:
- Check image quality and lighting
- Verify puzzle grids are clearly visible
- Adjust CLAHE parameters in `PuzzleProcessor.__init__`

**Tile Assembly Errors**:
- Ensure tiles are correctly extracted in Stage 1
- Try increasing iterations in simulated annealing
- Verify tile overlap/spacing expectations

**Memory Issues**:
- Reduce `restarts` or `iters` in simulated annealing
- Process images in batches instead of all at once

## 📜 License

This project is part of an image processing coursework/research project.

## 👤 Author

**Omar Tamer** - [GitHub](https://github.com/omartamerr)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📞 Support

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated**: May 2026
