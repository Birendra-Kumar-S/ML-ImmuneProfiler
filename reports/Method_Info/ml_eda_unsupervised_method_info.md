# Single-Cell RNA-seq Analysis: Comprehensive Workflow

## Overview

This analysis implements a detailed machine learning pipeline to classify cell types using single-cell RNA sequencing (scRNA-seq) data. The workflow includes preprocessing, exploratory data analysis (EDA), dimensionality reduction, unsupervised clustering, and an evaluation of the methods using metrics such as the Adjusted Rand Index (ARI) and Silhouette scores.

## Technical Background

Single-cell RNA-seq technology allows quantification of gene expression at the single-cell level, providing insight into cellular classification. Due to high dimensionality and sparsity inherent in scRNA-seq data, effective preprocessing and dimensionality reduction are crucial for meaningful downstream analyses.

## Data Preprocessing & EDA

### Data Loading and Quality Control

Data was loaded from a batch-corrected and normalized expression matrix dataset in R. Quality control involved:

- Removing unnamed columns
- Normalization and scaling of data to a standard fit
- Filtering based on quality metrics

### Feature Selection

Highly variable genes (HVGs) were identified based on variance analysis. This step is essential for focusing on genes that capture the biological variability rather than technical noise.

### Cell Type Distribution

The dataset contains:

- **76,893 cells**
- **14 distinct cell types**

Key statistics:

- Most abundant cell type: CD14 Mono (12.58%)
- Least abundant cell type: Treg (1.33%)
- Class imbalance ratio: **9.46:1** (most vs. least abundant)

### Gene Expression Feature Analysis

- Total features (genes): 1,999
- Sparsity: 17.42% zeros (typical for scRNA-seq data)
- Some of the top variable genes identified for downstream analyses include LYZ, HLA-DRA, S100A9, CD74, and GNLY.

## Dimensionality Reduction & Visualization

### Methods Chosen

- **Principal Component Analysis (PCA)**: Reduces dimensionality while retaining variance structure.
- **t-distributed Stochastic Neighbor Embedding (t-SNE)**: Useful for visualizing high-dimensional data in two dimensions.
- **Uniform Manifold Approximation and Projection (UMAP)**: Efficient at capturing both local and global data structure. (The best method for projection and visualization of such kind of data, {published backing through journals, also fits in with our inference from the plots and results.})

### Justifications

- PCA provides linear dimensionality reduction, valuable for initial feature extraction and visualizing feature spread.
- t-SNE and UMAP offer non-linear embeddings, beneficial for visualizing complex cell type structures.

### Results

- **UMAP** consistently provided better visualization and clearer cluster separations (highest Silhouette scores (can loosely say as cell separation in cluster score): 0.351 for 250 HVGs).
- PCA explained 90% variance with 19 PCs for 50 HVGs and 49 PCs for 100 HVGs.
- t-SNE and UMAP visualizations significantly improved cell-type separability compared to PCA alone.

## Unsupervised Clustering Analysis

### Algorithms Evaluated

- **K-means**
- **Hierarchical clustering** (Ward linkage with connectivity constraints)
- **Leiden and Louvain community detection algorithms**

### Clustering Performance Metrics

Metrics evaluated:

- **Adjusted Rand Index (ARI)**: measures similarity between clustering results and ground truth labels.
- **Silhouette Score**: assesses intra-cluster similarity and inter-cluster separation.

### Results Comparison

| Method       | HVGs | ARI        | Silhouette |
| ------------ | ---- | ---------- | ---------- |
| K-means      | 50   | 0.4824     | 0.1601     |
| Hierarchical | 50   | 0.4971     | **0.2222** |
| Leiden       | 50   | 0.4809     | 0.1821     |
| Louvain      | 50   | 0.4886     | 0.2107     |
| K-means      | 100  | **0.5067** | 0.0949     |
| Hierarchical | 100  | 0.4649     | 0.1117     |
| Leiden       | 100  | 0.4976     | 0.1161     |
| Louvain      | 100  | 0.5067     | 0.1202     |

### Interpretation

- **Best ARI (0.5067)**: K-means with 100 HVGs indicates a 50.7% overlap with ground truth labels. But Louvain and Leiden methods consistently lead with high values as well (the justification of using these two clustering methods is that, they are the cluster methods commonly used by many rna-seq data processing packages like Seurat in R and also many publications have reported consistent high values) 
- Hierarchical clustering with 50 HVGs had the highest silhouette score, indicating better internal cluster cohesion and separation.

## Cluster and Dimensionality Reduction Visualization

UMAP consistently demonstrated superior cell-type separation, making it the optimal projection method for interpreting clustering results visually.

### Why RNA-seq Data Forms Such Clusters

Cell types inherently exhibit distinct transcriptional profiles. scRNA-seq captures these unique expression signatures, resulting in clear groupings or clusters. Biological processes such as differentiation, cell cycle phases, and environmental responses also contribute to distinct clustering patterns. Hence these clusters have a very complex pattern and may have overlaps, hence on paper too they have shown to yield lower ARI and silhouette scores than other type of data. 

## Choosing Optimal Parameters for Downstream Tasks

- **Optimal HVGs**: 100 (balance of ARI and visual clarity) and also captures a lot of good variance without loosing any essential information
- **Optimal method**: K-means clustering due to its simplicity, high ARI, and computational efficiency.
- **Dimensionality Reduction**: UMAP recommended for visual and analytical clarity.

## Final Recommendations for Downstream Supervised Learning

- Use the top 100 HVGs for feature selection.
- Apply UMAP for dimensionality reduction to preserve local and global data structure.
- Employ K-means clustering results as initial labels or for further annotation refinement in supervised models.

### Assumptions and Limitations

- Assumed minimal batch effect post-correction.
- Cluster interpretations rely on known cell-type labels, which may not encompass all cellular heterogeneity. (Due to manual labelling, we might have missed some divergent labels and other intricacies)
- Unsupervised methods inherently depend on hyperparameter selection, influencing outcomes. Hence after an experiment with different HVG numbers and dimensionality reduction parameters, we have selected top 100 HVGs for our ML pipeline. There is always a trade-off here between variance, information, pattern and general biological context handler without overfitting the model. Finding a middle ground sweet spot without a lot of deviation in methodology is to be considered and what we have experimented with.

---





