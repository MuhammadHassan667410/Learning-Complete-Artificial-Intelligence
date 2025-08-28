# 🛒 Market Basket Analysis – Apriori & ECLAT

This project explores **Association Rule Learning** techniques to identify relationships between products in a retail dataset. By analyzing transaction data, we can uncover patterns like *"Customers who buy X are also likely to buy Y."*

---

## 📌 Project Overview
- Implemented **Apriori Algorithm** using the `apyori` library
- Generated **association rules** based on support, confidence, and lift
- Visualized **top rules** to better understand product relationships
- Applied **ECLAT Algorithm** as an alternative for mining frequent itemsets

---

## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib  
- Apyori (for Apriori algorithm)  

---

## 📊 Results
- Extracted frequent itemsets from 7500+ transactions  
- Identified **top 10 association rules** ranked by lift  
- Compared Apriori with **ECLAT** for performance and insights  

---

## 📂 Notebook Contents
1. Data Preprocessing  
2. Apriori Algorithm Implementation  
3. Association Rules Extraction  
4. Results DataFrame (Support, Confidence, Lift)  
5. ECLAT Algorithm  

---

## 🚀 How to Run
1. Clone the repo  
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib apyori
