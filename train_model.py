from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
NUMERIC = ["amount", "hour", "distance_km", "transactions_last_hour", "international"]
CATEGORICAL = ["merchant_category"]

def generate_reference_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    categories = ["Grocery","Fuel","Shopping","Travel","Restaurant",
                  "Electronics","Online Services","ATM","Healthcare"]
    return pd.DataFrame({
        "amount": np.clip(rng.lognormal(7.1,.75,n),80,25000).round(2),
        "hour": rng.integers(7,23,n),
        "merchant_category": rng.choice(categories,n,p=[.19,.12,.14,.08,.16,.07,.09,.06,.09]),
        "distance_km": np.clip(rng.gamma(2,2.2,n),0,22).round(2),
        "transactions_last_hour": rng.choice([1,2,3],n,p=[.72,.23,.05]),
        "international": rng.choice([0,1],n,p=[.985,.015])
    })

def main():
    reference = generate_reference_data()
    pre = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocessor", pre),
        ("model", IsolationForest(n_estimators=200, contamination="auto",
                                  random_state=42, n_jobs=-1))
    ])
    model.fit(reference[NUMERIC+CATEGORICAL])
    scores = -model.decision_function(reference[NUMERIC+CATEGORICAL])
    bundle = {"model":model, "reference_scores":scores,
              "threshold":float(np.quantile(scores,.95)),
              "features":NUMERIC+CATEGORICAL,
              "training_rows":len(reference)}
    (ROOT/"models").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT/"models"/"sentinelai_model.joblib")
    print("Saved model bundle:", ROOT/"models"/"sentinelai_model.joblib")
    print("Training rows:",len(reference)," | Threshold:",bundle["threshold"])

if __name__ == "__main__":
    main()
