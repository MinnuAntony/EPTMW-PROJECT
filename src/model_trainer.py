# model_trainer.py content

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib

def train_models(data_file, model_prefix):
    """Train classification, throughput, and energy efficiency models."""
    df = pd.read_csv(data_file)
    
    # Outlier detection using Isolation Forest
    iso = IsolationForest(contamination=0.05)
    df = df[iso.fit_predict(df.drop(columns=['power'])) == 1]
    
    # Normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.drop(columns=['power']))
    y = df['power']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train workload classification model (XGBoost)
    classification_model = xgb.XGBClassifier(eval_metric='mlogloss')
    classification_model.fit(X_train, y_train)
    joblib.dump(classification_model, f'{model_prefix}_classification.pkl')

    # Train throughput and efficiency models (ensemble: XGBoost, LightGBM, CatBoost)
    models = {
        'xgb': xgb.XGBRegressor(),
        'lgb': lgb.LGBMRegressor(),
        'cat': cb.CatBoostRegressor(verbose=0)
    }
    predictions = [model.fit(X_train, y_train).predict(X_test) for model in models.values()]
    mae = mean_absolute_error(y_test, np.mean(predictions, axis=0))
    print(f"Throughput Model Ensemble MAE: {mae}")
    
    joblib.dump(models, f'{model_prefix}_throughput_ensemble.pkl')

if __name__ == "__main__":
    train_models('data/metrics.csv', 'data/eptmw_model')

