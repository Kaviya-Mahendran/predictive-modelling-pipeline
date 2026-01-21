import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(path: str):
    """
    Load privacy-safe feature data and prepare it for modelling.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """

    df = pd.read_csv(path)

    # -------------------------------
    # Target variable
    # -------------------------------
    y = df["churn_risk"]

    # -------------------------------
    # Drop non-feature columns
    # -------------------------------
    X = df.drop(columns=["churn_risk", "customer_id_hash"])

    # -------------------------------
    # One-hot encode categoricals
    # -------------------------------
    categorical_cols = [
        "engagement_band",
        "signup_channel_group",
        "activity_trend_30d",
    ]

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    feature_names = X.columns.tolist()

    # -------------------------------
    # Train / test split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # -------------------------------
    # Scale numeric features
    # -------------------------------
    numeric_cols = [
        "activity_recency_days",
        "transactions_90d",
        "total_spend_90d",
        "avg_transaction_value",
    ]

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, feature_names
