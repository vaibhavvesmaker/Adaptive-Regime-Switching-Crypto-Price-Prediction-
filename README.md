# Adaptive Regime-Switching Crypto Price Prediction 🚀  

## **Overview**  
This project builds a **real-time crypto market prediction system** using a **CNN-LSTM hybrid model**. It forecasts **market regimes (bullish, bearish, neutral)** and predicts short-term price movements (**30m, 1h, 4h windows**) to enhance algorithmic trading strategies.  

## **📊 Key Features**  
✅ **Market Regime Classification:** Predicts whether the market is bullish, bearish, or neutral.  
✅ **Time-Series Forecasting:** Short-term price predictions using **CNN-LSTM**.  
✅ **Feature Engineering:**  
   - Technical indicators: RSI, MACD, Bollinger Bands, Fibonacci levels  
   - Volume & liquidity analysis: Bid-ask spread, VWAP  
   - Sentiment analysis: Twitter & Reddit NLP scores  
   - On-chain metrics: BTC dominance, open interest, funding rates  
✅ **Backtesting & Strategy Validation:**  
   - Uses **Backtrader** to simulate trading strategies  
   - Evaluates **Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD)**  
✅ **Live Deployment:**  
   - Deployed via **AWS SageMaker** with FastAPI  
   - Integrated with **Kafka for real-time data streaming**  
✅ **Monitoring & Model Drift Detection:**  
   - **Population Stability Index (PSI)** for feature distribution monitoring  
   - Automated retraining when drift exceeds 0.2 threshold  

---

## **🛠️ Tech Stack**
### **Languages & Libraries**  
🔹 **Python** – Main development language  
🔹 **TensorFlow/Keras** – Deep learning framework  
🔹 **Scikit-learn & XGBoost** – Feature engineering & baseline models  
🔹 **Backtrader** – Backtesting framework for strategy validation  
🔹 **AWS SageMaker** – Model training & deployment  
🔹 **FastAPI** – API for real-time inference  
🔹 **Kafka** – Streaming real-time market data  
🔹 **Power BI / Tableau** – Data visualization for insights  

---

## **📂 Project Structure**
```plaintext
├── data/                   # Raw & processed datasets
├── models/                 # Trained models & checkpoints
├── notebooks/              # Jupyter Notebooks for EDA & training
├── scripts/                # Python scripts for preprocessing, training, inference
│   ├── train_model.py      # CNN-LSTM training script
│   ├── preprocess_data.py  # Data cleaning & feature engineering
│   ├── deploy_api.py       # FastAPI deployment script
│   ├── monitor_model.py    # Drift detection & retraining
├── backtesting/            # Trading strategy validation
├── README.md               # Project documentation
