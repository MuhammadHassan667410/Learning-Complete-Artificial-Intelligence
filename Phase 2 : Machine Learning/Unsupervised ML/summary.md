# 🔍Unsupervised Learning – Clustering & Dimensionality Reduction

In this segment, I explored **Unsupervised Learning**, where models uncover hidden patterns or groupings in unlabeled data.  
The focus was on **clustering algorithms** and **dimensionality reduction** techniques, essential for data exploration and feature engineering.

---

## 🧠 What is Unsupervised Learning?

Unsupervised learning models:
- Work without labels or known outputs
- Aim to find structure in data
- Common tasks: **clustering**, **compression**, **association mining**

---

## 📅 Daily Breakdown

### 📌 **K-Means Clustering**
- Partitioned data into `k` clusters by minimizing intra-cluster distance
- Concepts: Centroids, Inertia, Elbow Method, Random Initialization
- Implemented with `scikit-learn` and visualized results
- Strengths: Simple and fast  
- Limitations: Sensitive to `k` and outliers

---

### 📌 **Hierarchical Clustering**
- Built nested clusters using **agglomerative** method
- Concepts: Dendrograms, Linkage Criteria (ward, complete, average)
- Used `scipy` and `scikit-learn` to perform clustering and plot trees
- Best for small to medium-sized datasets

---

### 📌 **DBSCAN (Density-Based Clustering)**
- Density-based approach to discover arbitrary-shaped clusters
- Parameters: `eps`, `min_samples`
- Automatically detects noise and outliers
- Compared DBSCAN vs K-Means on non-spherical data

---

### 📌 **Clustering on Real Datasets + Summary**
- Applied K-Means, Hierarchical, and DBSCAN on datasets like:
  - Iris Dataset
  - Mall Customer Segmentation
- Compared performance and suitability of each algorithm
- Wrapped up unsupervised learning phase with practical use cases

---

## ✅ Outcome

By the end, I could:
- Use clustering to explore structure in unlabeled data
- Choose appropriate algorithms for different data distributions
- Visualize clusters and evaluate separability
- Understand when to apply each unsupervised method

---
