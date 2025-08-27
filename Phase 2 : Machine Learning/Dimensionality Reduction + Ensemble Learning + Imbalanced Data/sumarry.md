# 📊Dimensionality Reduction, Ensemble Learning & Imbalanced Data

This part of my AI journey dives into **advanced machine learning concepts** — focusing on reducing complexity, improving model performance through ensembles, and addressing real-world data imbalance.

---

## 🔻 Principal Component Analysis (PCA)

- Unsupervised technique for **dimensionality reduction**
- Projects high-dimensional data to a lower-dimensional space
- Concepts:
  - Covariance matrix
  - Eigenvectors & eigenvalues
  - Explained variance ratio
- Applications: Visualization, Speed-up model training, Noise reduction
- Implemented using `scikit-learn` and visualized 2D projections

---

## 🎯 Feature Selection Techniques

Learned how to select the most relevant features for training:

- **Variance Threshold**: Remove low-variance features
- **Chi-Squared (χ²) Test**: For categorical input features
- **Mutual Information**: Measure dependency between features and target
- Compared feature selection vs dimensionality reduction

---

## 🧺 Bagging – Bootstrap Aggregation

- Combines predictions from multiple models trained on random subsets
- Reduces variance and improves generalization
- Explored:
  - Bootstrapped sampling
  - Random Forest: an ensemble of decision trees
- Key libraries: `scikit-learn` (`RandomForestClassifier`, `BaggingClassifier`)

---

## 🚀Boosting – AdaBoost, Gradient Boosting, XGBoost

- Ensemble technique that **builds models sequentially**, each correcting the errors of the previous
- Explored:
  - **AdaBoost**: Focuses on misclassified samples
  - **Gradient Boosting**: Optimizes loss via gradients
  - **XGBoost**: Scalable, regularized boosting system
- Compared accuracy, training time, and overfitting risks

---

## 🧠Stacking (Stacked Generalization)

- Combines multiple base learners using a **meta-learner**
- Each model contributes its prediction to train a final model
- Implemented a stacked model using:
  - Logistic Regression + Random Forest + XGBoost as base models
  - Logistic Regression as the meta-model
- Great for squeezing extra performance

---

## ⚖️ Handling Imbalanced Datasets

Real-world datasets often suffer from class imbalance (e.g., fraud detection).

Covered techniques:
- **SMOTE** (Synthetic Minority Oversampling Technique)
- **Random Undersampling**
- **Class Weights** in loss function
- Visualized the effect of each method on model performance
- Used `imblearn`, `scikit-learn`, `matplotlib`

---

## ✅ Outcome

By the end, I was able to:
- Reduce dimensionality for faster and better modeling
- Choose and tune ensemble models for high performance
- Address class imbalance for fairer and more accurate predictions

---
