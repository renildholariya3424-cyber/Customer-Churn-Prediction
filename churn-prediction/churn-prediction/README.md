# 📉 Customer Churn Prediction & Deployment

Predicts whether a telecom customer will leave (churn), so a business can offer retention deals before it happens.
The project covers the full ML workflow: data cleaning, preprocessing, comparing three models, and serving the best
one through a **FastAPI** endpoint with a **Streamlit** front end.

## Workflow
```
Telco data → Cleaning → Preprocessing (scaling + one-hot) → LogReg vs Random Forest vs XGBoost
→ 5-fold CV + test metrics → Best model saved → FastAPI /predict → Streamlit UI
```

## Key ML decisions
- **No data leakage:** scaling and encoding are inside a scikit-learn `Pipeline`, so they are fit on training data only.
- **Class imbalance** (~26% churn): handled with `class_weight="balanced"` and XGBoost `scale_pos_weight`.
- **Metrics:** precision, recall, F1 and ROC-AUC instead of accuracy, because accuracy is misleading on imbalanced data.
- **Model selection** uses cross-validation F1, keeping the test set as an unbiased final check.
- `TotalCharges` is blank for new customers (tenure 0), so it is converted to numeric and filled with 0.

## Results
Run `python train.py` and paste the printed table here:

| Model | CV F1 | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · FastAPI · Pydantic · Streamlit

## Project Structure
```
churn-prediction/
├── data/              # put the Kaggle CSV here
├── model/             # created by train.py (model + metrics.json)
├── train.py           # preprocessing, training, evaluation, model saving
├── api.py             # FastAPI prediction endpoint
├── app.py             # Streamlit UI
└── requirements.txt
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Download the dataset from Kaggle (`blastchar/telco-customer-churn`) and put
`WA_Fn-UseC_-Telco-Customer-Churn.csv` inside the `data/` folder.

## Run
```bash
python train.py                # trains 3 models, prints results, saves the best
uvicorn api:app --reload       # terminal 1: API at http://localhost:8000/docs
streamlit run app.py           # terminal 2: UI at http://localhost:8501
```

## API
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Status and whether the model is loaded |
| GET | `/model-info` | Model comparison results and top features |
| POST | `/predict` | Customer details → churn probability, prediction, risk level |

## Author
**Renil Dholariya** – [GitHub](https://github.com/renildholariya3424-cyber) · [LinkedIn](https://linkedin.com/in/renil-dholariya-023806372)
