
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CLASSIFICATION_MODEL_FILE = "data/workload_classification_model.pkl"
THROUGHPUT_MODEL_FILE = "data/workload_throughput_model.pkl"

def train_workload_models(data_file):
    df = pd.read_csv(data_file)
    X = df.drop(columns=['workload_type', 'throughput'])
    y_class = df['workload_type']
    y_throughput = df['throughput']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_class, test_size=0.2)
    classification_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    classification_model.fit(X_train_c, y_train_c)
    joblib.dump(classification_model, CLASSIFICATION_MODEL_FILE)
    print(f"Workload classification model saved to {CLASSIFICATION_MODEL_FILE}")
    
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_scaled, y_throughput, test_size=0.2)
    throughput_model = xgb.XGBRegressor()
    throughput_model.fit(X_train_t, y_train_t)
    joblib.dump(throughput_model, THROUGHPUT_MODEL_FILE)
    print(f"Workload throughput model saved to {THROUGHPUT_MODEL_FILE}")

def detect_workload(performance_counters):
    classification_model = joblib.load(CLASSIFICATION_MODEL_FILE)
    throughput_model = joblib.load(THROUGHPUT_MODEL_FILE)
    
    workload_type = classification_model.predict([performance_counters])[0]
    predicted_throughput = throughput_model.predict([performance_counters])[0]

    known_profiles = {
        "CPU-intensive": [1, 0, 0],
        "Memory-intensive": [0, 1, 0],
        "Disk-intensive": [0, 0, 1]
    }

    unknown_features = np.array(performance_counters).reshape(1, -1)
    similarities = {}
    for workload, profile in known_profiles.items():
        similarity = cosine_similarity(unknown_features, np.array(profile).reshape(1, -1))[0][0]
        similarities[workload] = similarity
    
    total_similarity = sum(similarities.values())
    weight_vector = {k: v / total_similarity for k, v in similarities.items()}
    
    print(f"Detected workload type: {workload_type}")
    print(f"Predicted throughput: {predicted_throughput}")
    print(f"Workload weight vector: {weight_vector}")

    return workload_type, predicted_throughput, weight_vector

if __name__ == "__main__":
    print("Training workload models using XGBoost...")
    train_workload_models("data/performance_counters.csv")
    sample_counters = [0.8, 0.1, 0.05, 100, 200, 50]
    detect_workload(sample_counters)
