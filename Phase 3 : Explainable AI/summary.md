# 📊 Phase 3: Explainable AI=

In this phase, I explored **model interpretability techniques** to understand how and why machine learning models make predictions. Accuracy alone is not enough in real-world applications — transparency and trust are equally important.

---

## ✅  SHAP (SHapley Additive exPlanations)
- Learned about **Shapley values** from game theory and how they apply to model interpretability.  
- Understood the difference between **global vs. local interpretability**.  
- Explored different SHAP approaches:
  - **KernelSHAP** (model-agnostic)  
  - **TreeSHAP** (optimized for tree-based models like Random Forest, XGBoost, LightGBM)  
  - **DeepSHAP** (for neural networks)  
- Implemented SHAP with XGBoost and created multiple plots:
  - **Summary plot (beeswarm & bar)** → overall feature importance  
  - **Dependence plot** → feature value vs. impact  
  - **Waterfall & Force plots** → explanation for single predictions  
  - **Decision & Heatmap plots** → feature interactions across many samples  

**Key Insight:**  
> SHAP not only tells *which* features are important, but also *how* they influence predictions (positive/negative contribution).

---

## ✅LIME (Local Interpretable Model-Agnostic Explanations)
- Learned how **LIME** explains individual predictions by approximating the model locally with simpler, interpretable models.  
- Compared **LIME vs. SHAP**:
  - LIME → fast, simple, local explanations.  
  - SHAP → mathematically consistent, global + local explanations.  
- Applied LIME on a classification model to visualize how features contributed to a single prediction.  
- Understood strengths and limitations:
  - Pros → easy to implement, works with any black-box model.  
  - Cons → instability (different runs may give slightly different explanations).  

**Key Insight:**  
> LIME gives an intuitive, easy-to-understand explanation of single predictions — but SHAP is more robust and consistent for both local and global interpretability.

---

### 📌 Phase 3 Takeaway
Explainable AI is crucial in domains like **healthcare, finance, and legal AI**, where models must not only perform well but also be **transparent, interpretable, and trustworthy**.  

---
