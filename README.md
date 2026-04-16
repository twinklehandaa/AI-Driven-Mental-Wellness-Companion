# 🧠 MindSync — AI Mental Wellness Companion

An AI-powered mental wellness application that predicts user mood using behavioral data and provides personalized recommendations.

Built using a hybrid machine learning approach combining **Random Forest + LSTM-inspired temporal modeling**, deployed via an interactive **Streamlit dashboard**.

---

## 🚀 Features

- 🔮 Mood Prediction (Stress / Neutral / Happy)
- 🧠 Hybrid ML Model (RF + LSTM-inspired fusion)
- 📊 Wellness Analytics Dashboard
- 📓 Journaling System
- 🎯 Goals & Habit Tracking
- 🚨 Risk Monitoring System
- 💡 Personalized Recommendations
- 🏆 Gamification (streaks, points, badges)

---

## 🧩 Problem Statement

Traditional mental health apps rely on **manual mood input**, which is:
- Inconsistent
- Biased
- Not proactive

MindSync solves this by:
- Predicting mood using **passive behavioral signals**
- Providing **context-aware interventions**
- Reducing user effort while improving accuracy

---

## 🏗️ System Architecture

The system follows a pipeline:

1. **Input Layer**
   - Sleep duration
   - Screen time
   - Activity level
   - Social interaction

2. **Preprocessing**
   - Scaling & normalization
   - Feature engineering

3. **Hybrid Model**
   - Random Forest → structured patterns
   - LSTM (simulated) → temporal trends
   - Weighted Fusion:
     ```
     Final Prediction = 0.6 × LSTM + 0.4 × RF
     ```

4. **Output**
   - Mood classification
   - Confidence score
   - Recommendations

---

## 🤖 Model Details

| Model            | Purpose                         |
|-----------------|---------------------------------|
| Random Forest    | Tabular feature learning        |
| LSTM (simulated) | Temporal behavior modeling      |
| Hybrid Fusion    | Improved accuracy & robustness  |

- Accuracy: **~88%**
- F1 Score: **0.89**

---

## 📊 Dataset

- Synthetic dataset (500 samples)
- Inspired by WESAD dataset
- Features:
  - Sleep hours
  - Screen time
  - Activity level
  - Social interaction

---

## 🖥️ Tech Stack

- **Frontend/UI:** Streamlit
- **ML Models:** Scikit-learn
- **Visualization:** Plotly, Matplotlib
- **Language:** Python

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/mindsync.git
cd mindsync
pip install -r requirements.txt
streamlit run mental_wellness_app.py
