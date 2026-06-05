"""
feature_engineering.py
-----------------------
Custom feature engineering for the Meridian Financial complaint escalation model.

This file must be in the same folder as 72541_Pipeline.pkl when loading the model.
Usage in a new notebook:
    from feature_engineering import prepare_features
    X_val = prepare_features(df_val)
    predictions = loaded_pipeline.predict(X_val)
"""

import pandas as pd


# These must match EXACTLY what was defined in the training notebook
CATEGORICAL_FEATURES = [
    'Product',
    'Sub-product',
    'Issue',
    'Submitted via',
    'State',
    'Tags',
    'Timely response?',
    'Company response to consumer',
]

NUMERICAL_FEATURES = [
    'days_to_send',
    'has_narrative',
    'month',
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def prepare_features(df):
    """
    Apply feature engineering to a raw complaints dataframe.

    Creates three engineered features from the raw columns, then returns
    only the columns the model was trained on.

    Parameters
    ----------
    df : pd.DataFrame
        Raw complaints dataframe. Must contain the same columns as the
        training data (Date received, Date sent to company,
        Consumer complaint narrative, plus all categorical features).

    Returns
    -------
    pd.DataFrame
        Dataframe with only the model's input features, ready to pass
        directly to loaded_pipeline.predict().
    """
    df_out = df.copy()

    # Feature 1: Days from complaint receipt to being sent to the company
    df_out['Date received']        = pd.to_datetime(df_out['Date received'],        errors='coerce')
    df_out['Date sent to company'] = pd.to_datetime(df_out['Date sent to company'], errors='coerce')
    df_out['days_to_send'] = (df_out['Date sent to company'] - df_out['Date received']).dt.days
    df_out['days_to_send'] = df_out['days_to_send'].clip(lower=0)

    # Feature 2: Whether the consumer submitted a complaint narrative (1 = yes, 0 = no)
    df_out['has_narrative'] = df_out['Consumer complaint narrative'].notna().astype(int)

    # Feature 3: Month the complaint was received
    df_out['month'] = df_out['Date received'].dt.month

    return df_out[ALL_FEATURES]
