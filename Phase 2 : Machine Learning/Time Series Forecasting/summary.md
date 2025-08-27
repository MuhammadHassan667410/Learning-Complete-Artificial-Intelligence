# ⏳ Time Series Forecasting

This section is part of my **AI Learning Journey**.  
Over these 3 days, I explored the foundations and advanced techniques of **Time Series Analysis & Forecasting**.  

---

## 📅  Time Series Basics
- Learned what makes **time series** data unique.  
- Key concepts: **Trend, Seasonality, Stationarity, Lag Features**.  
- Hands-on with the **Airline Passengers dataset**:  
  - Visualized long-term **trend** and **seasonality**.  
  - Applied **ADF Test** → confirmed non-stationarity.  
  - Used **differencing (1st, 2nd, seasonal)** to achieve stationarity.  
  - Created **lag features** for predictive modeling.  

---

## 📅Forecasting Methods
- Explored **classical forecasting models**:  
  - **ARIMA** (AutoRegressive Integrated Moving Average)  
  - **SARIMA** for seasonal adjustments  
  - **Facebook Prophet** for trend + seasonality forecasting  
- Compared results of different models.  
- Learned how model assumptions (stationarity, seasonality) impact predictions.  

---

## 📅  Deep Learning for Time Series
- Introduced **neural network approaches**:  
  - **RNNs (Recurrent Neural Networks)** for sequential data.  
  - **LSTM (Long Short-Term Memory)** & **GRU (Gated Recurrent Units)** → handle long-term dependencies.  
- Built a simple **LSTM model** for forecasting passenger data.  
- Compared deep learning performance with ARIMA/Prophet.  

---

## 📊 Key Takeaways
- Time series data needs **stationarity checks** and **feature engineering** before modeling.  
- **Classical models (ARIMA/Prophet)** are interpretable and great for short/mid-term forecasting.  
- **Deep learning models (LSTM/GRU)** shine for complex, long-range dependencies.  
- Choosing the right method depends on dataset size, complexity, and goals.  

---
