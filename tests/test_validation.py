import pandas as pd
from app import validate_frame

def test_valid_data():
    df = pd.DataFrame([{
        "amount": 1200, "hour": 14, "merchant_category": "Grocery",
        "distance_km": 2, "transactions_last_hour": 1, "international": 0
    }])
    assert len(validate_frame(df)) == 1

def test_missing_column():
    df = pd.DataFrame([{"amount": 100, "hour": 12}])
    try:
        validate_frame(df)
    except ValueError as e:
        assert "Missing required columns" in str(e)
    else:
        raise AssertionError("Expected missing-column validation error")
