# Handwritten Digit Clustering (DBSCAN + PCA)

This project aims to group similar handwritten digits from the MNIST dataset without using their labels and visualize the clustering process. It reduces the high-dimensional image data using PCA and applies DBSCAN clustering.

## Structure
- `data/`: Contains the MNIST dataset (`mnist_train.csv`).
- `src/`: Contains python scripts for preprocessing, PCA, DBSCAN, and visualization.
- `notebooks/`: Contains the main Jupyter notebook for interactive exploration.
- `outputs/`: Output directory for generated plots and metrics.

## Setup
1. Create a virtual environment and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Jupyter Notebook to see the step-by-step process and visualizations:
   ```bash
   jupyter notebook notebooks/digit_clustering.ipynb
   ```
