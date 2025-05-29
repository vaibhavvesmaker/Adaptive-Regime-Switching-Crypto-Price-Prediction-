### **📌 Adaptive Regime-Switching Crypto Price Prediction**  

```markdown 
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
```

---

## **🚀 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/crypto-regime-prediction.git
cd crypto-regime-prediction
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Data Preprocessing**
```bash
python scripts/preprocess_data.py
```

### **4️⃣ Train the Model**
```bash
python scripts/train_model.py
```

### **5️⃣ Deploy the API**
```bash
uvicorn scripts.deploy_api:app --host 0.0.0.0 --port 8000
```

---

## **🔬 Model Architecture**
- **CNN-LSTM Hybrid**  
  - **CNN** extracts localized patterns from multi-feature input tensors.  
  - **LSTM** captures sequential dependencies in crypto price movements.  
  - **Attention Mechanism** dynamically weights key market features.  

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Attention
from tensorflow.keras.models import Sequential

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, features)),
    BatchNormalization(),
    LSTM(128, return_sequences=True),
    Attention(),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Bullish, Bearish, Neutral
])
```

---

## **🛠️ Monitoring & Drift Detection**
📌 **Drift Detection Methods:**  
- **Population Stability Index (PSI):** Triggers retraining if PSI > 0.2.  
- **Jensen-Shannon Divergence:** Identifies feature distribution shifts.  
- **Kolmogorov-Smirnov Test:** Checks probability distributions over time.  

```python
if psi_value > 0.2:
    retrain_model()
```

---

## **📈 Performance Metrics**
- **AUC-ROC & Log Loss:** Evaluate market regime classification performance.  
- **Sharpe Ratio & Sortino Ratio:** Validate trading strategy effectiveness.  
- **Max Drawdown (MDD):** Assess downside risk.  

---

## **🛠 Future Improvements**
📌 **Enhancements in Progress:**  
- **Transformer-based models (TFT, Informer)** for improved sequential learning.  
- **Reinforcement Learning (RL)** to optimize trading execution strategies.  
- **Enhanced sentiment analysis** using LLMs for deeper NLP insights.  

---

## **📢 Contribution Guidelines**
1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature-new-model`  
3. Commit changes: `git commit -m "Added new preprocessing step"`  
4. Push to GitHub: `git push origin feature-new-model`  
5. Submit a Pull Request.  

---

## **📩 Contact & Support**
💬 **Author:** Vaibhav Vesmaker  


---

## **📜 License**
This project is licensed under **MIT License**. Feel free to use and modify.  

---

### **🚀 Ready to Trade Smarter?**
Start by running the model and integrating it into your **crypto trading strategies**! 🚀📈

```bash
python scripts/deploy_api.py
```

