# SentinelAI — AI-Powered Fraud Detection System

**GENAI CIA2 | Topic 3 | Academic prototype**

SentinelAI is a Streamlit web application that demonstrates transaction anomaly detection using an Isolation Forest machine-learning model.

## Features
- Dashboard with analysed, normal and suspicious transaction counts
- Single-transaction AI scanner
- CSV batch analysis with validation
- Session history, filtering and CSV export
- Model methodology and responsible-AI limitations
- Synthetic sample transaction dataset

## Technology
Python, Streamlit, pandas, NumPy, scikit-learn, Plotly and joblib.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install dependencies:

   `pip install -r requirements.txt`

5. Start the website:

   `streamlit run app.py`

The app opens in your browser, usually at `http://localhost:8501`.

## Sample CSV format
Required columns:
`amount,hour,merchant_category,distance_km,transactions_last_hour,international`

An optional `transaction_id` column is also supported.

## Machine-learning approach
The model is an unsupervised Isolation Forest trained on generated synthetic normal transaction patterns. Numerical columns are standardised and merchant category is one-hot encoded. An anomaly score is compared with the 95th percentile of scores from the synthetic reference dataset.

## Important limitations
This is an academic demonstration, not a production banking system. Synthetic data does not establish real-world accuracy. An anomaly is not proof of fraud. Do not use this application to block or approve real transactions.

## Project structure
- `app.py` — Streamlit web application
- `train_model.py` — model training and serialization
- `data/sample_transactions.csv` — synthetic sample data
- `models/` — generated model bundle (created by train_model.py)
- `Project_Report.docx` — project report
- `Presentation.pptx` — presentation
- `demo_script.md` — practical demonstration guide
- `requirements.txt` — dependencies
