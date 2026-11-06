
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "sample_transactions.csv"

NUMERIC = [
    "amount",
    "hour",
    "distance_km",
    "transactions_last_hour",
    "international"
]

CATEGORICAL = ["merchant_category"]

REQUIRED = NUMERIC + CATEGORICAL

st.set_page_config(
    page_title="SentinelAI | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def generate_reference_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)

    categories = [
        "Grocery", "Fuel", "Shopping", "Travel",
        "Restaurant", "Electronics", "Online Services",
        "ATM", "Healthcare"
    ]

    return pd.DataFrame({
        "amount": np.clip(
            rng.lognormal(7.1, 0.75, n), 80, 25000
        ).round(2),
        "hour": rng.integers(7, 23, n),
        "merchant_category": rng.choice(
            categories,
            n,
            p=[0.19, 0.12, 0.14, 0.08, 0.16,
               0.07, 0.09, 0.06, 0.09]
        ),
        "distance_km": np.clip(
            rng.gamma(2, 2.2, n), 0, 22
        ).round(2),
        "transactions_last_hour": rng.choice(
            [1, 2, 3], n, p=[0.72, 0.23, 0.05]
        ),
        "international": rng.choice(
            [0, 1], n, p=[0.985, 0.015]
        )
    })


def train_model(reference):
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore"),
         CATEGORICAL)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", IsolationForest(
            n_estimators=200,
            contamination="auto",
            max_samples="auto",
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(reference[REQUIRED])

    reference_scores = -pipeline.decision_function(
        reference[REQUIRED]
    )

    threshold = float(np.quantile(reference_scores, 0.95))

    return pipeline, reference_scores, threshold


@st.cache_resource
def load_model():
    reference = generate_reference_data()
    return train_model(reference)


def validate_frame(df):
    missing = [c for c in REQUIRED if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    data = df[REQUIRED].copy()

    for col in NUMERIC:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data["merchant_category"] = (
        data["merchant_category"].astype(str)
    )

    if data[NUMERIC].isna().any().any():
        raise ValueError(
            "Numeric columns contain blank or non-numeric values."
        )

    if (data["amount"] <= 0).any():
        raise ValueError("Amount must be greater than zero.")

    if not data["hour"].between(0, 23).all():
        raise ValueError("Hour must be between 0 and 23.")

    if (
        (data["distance_km"] < 0).any()
        or (data["transactions_last_hour"] < 1).any()
    ):
        raise ValueError(
            "Distance cannot be negative and transaction "
            "count must be at least 1."
        )

    if not data["international"].isin([0, 1]).all():
        raise ValueError("International must be 0 or 1.")

    return data


def explain_row(row):
    reasons = []

    if row["amount"] > 30000:
        reasons.append("Transaction amount is unusually high.")

    if row["hour"] < 6 or row["hour"] > 22:
        reasons.append(
            "Transaction occurred outside common daytime/evening hours."
        )

    if row["distance_km"] > 30:
        reasons.append(
            "Transaction location is far from the usual location."
        )

    if row["transactions_last_hour"] >= 4:
        reasons.append(
            "Several transactions occurred within one hour."
        )

    if row["international"] == 1:
        reasons.append(
            "Transaction is marked as international."
        )

    if not reasons:
        reasons.append(
            "No single extreme input was found; the model "
            "evaluated the combined feature pattern."
        )

    return reasons


def score_transactions(data, model, reference_scores, threshold):
    features = data[REQUIRED]

    anomaly_scores = -model.decision_function(features)

    risk = np.array([
        (reference_scores <= score).mean() * 100
        for score in anomaly_scores
    ])

    result = data.copy()
    result["anomaly_score"] = anomaly_scores
    result["risk_indicator"] = np.round(risk, 1)

    result["status"] = np.where(
        anomaly_scores >= threshold,
        "Suspicious",
        "Normal"
    )

    return result


if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()


model, reference_scores, threshold = load_model()


st.markdown("""
<style>
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

div[data-testid="stMetric"] {
    background: rgba(100,120,150,.08);
    padding: 14px;
    border-radius: 12px;
}

.hero {
    padding: 22px 24px;
    border-radius: 16px;
    color: white;
    background: linear-gradient(115deg,#10243c,#1768a7);
}

.small-muted {
    opacity: .78;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# Sidebar navigation
with st.sidebar:
    st.markdown("## 🛡️ SentinelAI")
    st.caption("AI-powered transaction anomaly detection")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "AI Scanner",
            "CSV Analysis",
            "Transaction History",
            "About the Model"
        ]
    )


# Main header
st.markdown(
    '<div class="hero">'
    '<h1 style="margin:0">🛡️ SentinelAI</h1>'
    '<div style="font-size:18px;margin-top:5px">'
    'Intelligent Fraud Detection Platform</div>'
    '<div class="small-muted" style="margin-top:10px">'
    'Machine learning • Anomaly detection • Transaction monitoring'
    '</div></div>',
    unsafe_allow_html=True
)


# OVERVIEW
if page == "Overview":

    st.subheader("Security overview")

    hist = st.session_state.history

    total = len(hist)

    suspicious = (
        int((hist["status"] == "Suspicious").sum())
        if total else 0
    )

    normal_count = total - suspicious

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions analysed", total)
    c2.metric("Normal", normal_count)
    c3.metric("Flagged", suspicious)

    c4.metric(
        "Flag rate",
        f"{(suspicious / total * 100):.1f}%"
        if total else "0.0%"
    )

    st.info(
        "Start with AI Scanner or upload a CSV to analyse "
        "transactions. The dashboard updates as you analyse data."
    )

    if total:

        left, right = st.columns(2)

        with left:
            counts = (
                hist["status"]
                .value_counts()
                .rename_axis("status")
                .reset_index(name="count")
            )

            st.plotly_chart(
                px.pie(
                    counts,
                    names="status",
                    values="count",
                    title="Detection distribution",
                    hole=0.55
                ),
                use_container_width=True
            )

        with right:
            st.plotly_chart(
                px.histogram(
                    hist,
                    x="risk_indicator",
                    color="status",
                    title="Anomaly indicator distribution",
                    nbins=15
                ),
                use_container_width=True
            )

        st.subheader("Latest analysed transactions")

        st.dataframe(
            hist.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.caption(
            "No user transactions have been analysed "
            "in this session yet."
        )


# AI SCANNER
elif page == "AI Scanner":

    st.subheader("Analyse one transaction")

    st.write(
        "Enter transaction details and let the Isolation Forest "
        "model evaluate the combined pattern."
    )

    with st.form("single_transaction"):

        a, b, c = st.columns(3)

        amount = a.number_input(
            "Amount (₹)",
            min_value=1.0,
            value=2500.0,
            step=100.0
        )

        hour = b.slider(
            "Transaction hour",
            0, 23, 14
        )

        category = c.selectbox(
            "Merchant category",
            [
                "Grocery", "Fuel", "Shopping", "Travel",
                "Restaurant", "Electronics", "Online Services",
                "ATM", "Healthcare"
            ]
        )

        d, e, f = st.columns(3)

        distance = d.number_input(
            "Distance from usual location (km)",
            min_value=0.0,
            value=3.0,
            step=1.0
        )

        frequency = e.number_input(
            "Transactions in last hour",
            min_value=1,
            value=1,
            step=1
        )

        international = f.selectbox(
            "International transaction",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        submitted = st.form_submit_button(
            "✦ Run AI analysis",
            type="primary",
            use_container_width=True
        )

    if submitted:

        one = pd.DataFrame([{
            "amount": amount,
            "hour": hour,
            "merchant_category": category,
            "distance_km": distance,
            "transactions_last_hour": frequency,
            "international": international
        }])

        scored = score_transactions(
            one, model, reference_scores, threshold
        )

        scored.insert(
            0,
            "transaction_id",
            f"MANUAL-{datetime.now().strftime('%H%M%S')}"
        )

        st.session_state.history = pd.concat(
            [st.session_state.history, scored],
            ignore_index=True
        )

        row = scored.iloc[0]

        st.divider()

        if row["status"] == "Suspicious":
            st.error(
                "Suspicious pattern detected — review recommended."
            )
        else:
            st.success(
                "No anomaly above the configured threshold was detected."
            )

        x, y = st.columns(2)

        x.metric(
            "Anomaly indicator",
            f'{row["risk_indicator"]:.1f}/100'
        )

        y.metric("Model status", row["status"])

        st.progress(
            float(row["risk_indicator"]) / 100
        )

        st.caption(
            "This indicator is an empirical percentile, "
            "not the probability that the transaction is fraudulent."
        )

        st.markdown("**Input observations**")

        for reason in explain_row(one.iloc[0]):
            st.write("• " + reason)


# CSV ANALYSIS
elif page == "CSV Analysis":

    st.subheader("Batch transaction analysis")

    st.write(
        "Upload a CSV containing the required transaction columns. "
        "A sample file is available below."
    )

    if not DATA_PATH.exists():
        st.error("Sample CSV file not found in the data folder.")
    else:
        sample = pd.read_csv(DATA_PATH)

        st.download_button(
            "Download sample CSV",
            sample.to_csv(index=False),
            "sentinelai_sample_transactions.csv",
            "text/csv"
        )

    uploaded = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded:

        try:
            raw = pd.read_csv(uploaded)

            clean = validate_frame(raw)

            scored = score_transactions(
                clean, model, reference_scores, threshold
            )

            if "transaction_id" in raw.columns:
                scored.insert(
                    0,
                    "transaction_id",
                    raw["transaction_id"].astype(str).values
                )
            else:
                scored.insert(
                    0,
                    "transaction_id",
                    [
                        f"UPLOAD-{i+1:04d}"
                        for i in range(len(scored))
                    ]
                )

            st.session_state.history = pd.concat(
                [st.session_state.history, scored],
                ignore_index=True
            )

            st.success(
                f"Successfully analysed {len(scored)} transactions."
            )

            st.dataframe(
                scored,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download analysis report CSV",
                scored.to_csv(index=False),
                "sentinelai_analysis_results.csv",
                "text/csv"
            )

            st.plotly_chart(
                px.histogram(
                    scored,
                    x="risk_indicator",
                    color="status",
                    title="Uploaded transaction risk distribution",
                    nbins=20
                ),
                use_container_width=True
            )

        except Exception as exc:
            st.error(f"Could not analyse this file: {exc}")


# TRANSACTION HISTORY
elif page == "Transaction History":

    st.subheader("Session transaction history")

    hist = st.session_state.history

    if hist.empty:

        st.info(
            "No transactions analysed yet. "
            "Use AI Scanner or CSV Analysis first."
        )

    else:

        filter_status = st.selectbox(
            "Filter status",
            ["All", "Normal", "Suspicious"]
        )

        view = (
            hist
            if filter_status == "All"
            else hist[hist["status"] == filter_status]
        )

        st.dataframe(
            view.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export full history CSV",
            hist.to_csv(index=False),
            "sentinelai_transaction_history.csv",
            "text/csv"
        )

        if st.button("Clear session history"):
            st.session_state.history = pd.DataFrame()
            st.rerun()


# ABOUT THE MODEL
elif page == "About the Model":

    st.subheader("How SentinelAI works")

    st.markdown("""
**Algorithm: Isolation Forest (unsupervised anomaly detection)**

1. **Reference data:** The application generates synthetic examples of ordinary transaction behaviour.
2. **Preprocessing:** Numerical features are standardised and merchant categories are one-hot encoded.
3. **Training:** 200 Isolation Trees are built using random feature selections and split points.
4. **Scoring:** Observations isolated in fewer tree splits receive more anomalous scores.
5. **Decision:** Scores above the 95th percentile of the synthetic reference scores are marked *Suspicious* for review.

**Features used**

- Transaction amount
- Hour of transaction
- Merchant category
- Distance from usual location
- Number of transactions in the last hour
- International transaction indicator

**Important limitations**

- The data is synthetic and does not represent actual bank transactions.
- An anomaly is not proof of fraud; legitimate unusual purchases may be flagged.
- The model may miss fraud that resembles normal activity.
- This academic prototype must not be used to approve, reject, or block real financial transactions.
""")

    st.code(
        "Preprocessing → Isolation Forest → Anomaly score → Threshold → Review",
        language="text"
    )

    st.metric("Isolation Forest trees", 200)

    st.metric(
        "Reference threshold",
        f"{threshold:.5f}"
    )

    st.caption(
        "Threshold is calibrated using the synthetic normal reference set."
    )
