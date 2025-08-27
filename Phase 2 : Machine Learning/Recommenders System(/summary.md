# Recommender Systems – Summary

##  Introduction & Content-Based Filtering
- **Recommender Systems Overview**  
  - Systems designed to suggest items (movies, books, products, etc.) to users.
  - Common applications: E-commerce, streaming platforms, social media.
  - Main types: Content-Based Filtering, Collaborative Filtering, Hybrid Systems.

- **Content-Based Filtering**  
  - Makes recommendations based on **item similarity** and a user’s past preferences.
  - Each item is described by a set of **features** (e.g., genre, director, price).
  - The system compares new items to those the user liked before.
  - Works well for users with **consistent tastes**.
  - Limitation: Can’t recommend very different types of items from what the user already consumes.

---

## Collaborative Filtering
- **Concept**  
  - Focuses on **similarities between users** or **between items**.
  - Predicts a user’s interest based on the interests of similar users.

- **Types**  
  - **User-User Collaborative Filtering**: Finds users with similar preferences and recommends what they liked.
  - **Item-Item Collaborative Filtering**: Finds items that are often liked together and recommends them.

- **Advantages**  
  - Can discover unexpected items (serendipity).
  - Doesn’t require item feature data.

- **Challenges**  
  - Cold Start Problem (new users or new items without ratings).
  - Scalability for large datasets.

---

## Hybrid Recommender Systems
- **Definition**  
  - Combines **Content-Based** and **Collaborative Filtering** to get the best of both.
  - Helps overcome limitations of individual methods.

- **Hybrid Approaches**  
  - **Weighted**: Combine scores from both systems with certain weights.
  - **Switching**: Use one method when data is sparse, the other otherwise.
  - **Feature Augmentation**: Use the output of one as input features for another.

- **Advantages**  
  - Reduces cold start problem.
  - Improves accuracy and diversity of recommendations.

---

##  Evaluation Metrics for Recommender Systems
- **Accuracy Metrics**
  - **RMSE (Root Mean Squared Error)** and **MAE (Mean Absolute Error)** for ratings prediction.
  - **Precision, Recall, F1-score** for top-N recommendations.

- **Ranking Metrics**
  - **MAP (Mean Average Precision)**: Measures precision at different cutoffs.
  - **NDCG (Normalized Discounted Cumulative Gain)**: Rewards highly ranked relevant items.

- **Beyond Accuracy**
  - **Diversity**: Avoid recommending only similar items.
  - **Serendipity**: Surprise users with unexpected yet relevant recommendations.
  - **Coverage**: Percentage of items the system can recommend.
