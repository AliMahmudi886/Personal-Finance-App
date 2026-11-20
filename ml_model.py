from contextlib import closing

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob

from models import create_connection


def fetch_expense_data(user_id):
    query = "SELECT date, amount, description FROM expenses WHERE user_id = ? ORDER BY date, id"
    with closing(create_connection()) as conn:
        df = pd.read_sql_query(query, conn, params=(user_id,))

    # Replace None with empty string in 'description' column
    df["description"] = df["description"].fillna("")

    # Ensure 'description' column is of type string
    df["description"] = df["description"].astype(str)

    return df


class InvalidExpenseHistory(ValueError):
    pass


def monthly_expenses(user_id):
    data = fetch_expense_data(user_id)[["date", "amount"]]
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    valid = data["date"].notna() & np.isfinite(data["amount"])
    if not valid.all():
        raise InvalidExpenseHistory
    data = data.set_index("date")
    if data.empty:
        return data
    monthly = data.resample("ME").sum()
    if not np.isfinite(monthly["amount"]).all():
        raise InvalidExpenseHistory
    return monthly


def train_lstm_model(user_id):
    data = monthly_expenses(user_id)

    if len(data) < 2:
        return None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[["amount"]])

    X, y = [], []
    for i in range(len(scaled_data) - 1):
        X.append(scaled_data[i])
        y.append(scaled_data[i + 1])
    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    from tensorflow.keras import Input, Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X, y, epochs=20, batch_size=1, verbose=0)

    return model, scaler


def predict_next_month_lstm(user_id, model, scaler):
    if not model or not scaler:
        return None

    data = monthly_expenses(user_id)

    if data.shape[0] < 1:
        return None

    scaled_last_value = scaler.transform(data[["amount"]].tail(1))
    scaled_last_value = np.reshape(scaled_last_value, (1, 1, 1))

    prediction = model.predict(scaled_last_value, verbose=0)
    prediction = scaler.inverse_transform(prediction)
    next_month_prediction = prediction[0, 0]

    return next_month_prediction


def detect_anomalies(user_id):
    data = monthly_expenses(user_id)

    if data.shape[0] < 2:
        return pd.DataFrame()

    model = IsolationForest(contamination=0.1)
    data["anomaly"] = model.fit_predict(data[["amount"]])
    anomalies = data[data["anomaly"] == -1].drop(columns=["anomaly"])

    return anomalies


def recommend_savings_plan(user_id):
    data = fetch_expense_data(user_id)
    if data.empty:
        return None

    if data.shape[0] < 2:
        return None

    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    if not np.isfinite(data["amount"]).all():
        raise InvalidExpenseHistory
    model = NearestNeighbors(n_neighbors=1)
    model.fit(data[["amount"]])

    recommendation = model.kneighbors(data[["amount"]].tail(1), return_distance=False)
    recommended_amount = data.iloc[recommendation[0][0]]["amount"]

    return recommended_amount


def train_autoencoder(user_id):
    data = monthly_expenses(user_id)

    if data.shape[0] < 2:
        return None, None

    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    input_dim = scaled_data.shape[1]
    encoding_dim = max(1, input_dim // 2)

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh")(input_layer)
    decoder = Dense(input_dim, activation="linear")(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(
        scaled_data, scaled_data, epochs=50, batch_size=8, shuffle=True, verbose=0
    )

    return autoencoder, scaler


def detect_anomalies_autoencoder(user_id, model, scaler):
    data = monthly_expenses(user_id)

    if data.shape[0] < 2:
        return None

    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data, verbose=0)
    mse = np.mean(np.power(scaled_data - predictions, 2), axis=1)
    anomaly_threshold = np.percentile(mse, 95)
    data["anomaly"] = mse > anomaly_threshold

    return data[data["anomaly"]]


def cluster_expenses(user_id, method="kmeans", n_clusters=3):
    if method not in {"kmeans", "hierarchical", "dbscan"} or not 2 <= n_clusters <= 20:
        raise ValueError("Invalid clustering settings.")
    data = fetch_expense_data(user_id)
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    tfidf = TfidfVectorizer(stop_words="english")
    analyzer = tfidf.build_analyzer()
    meaningful = (
        data["description"]
        .map(lambda description: bool(analyzer(description)))
        .astype(bool)
    )
    data = data.loc[meaningful & np.isfinite(data["amount"])].copy()
    minimum_rows = 2 if method == "dbscan" else n_clusters
    if len(data) < minimum_rows:
        return None
    tfidf_matrix = tfidf.fit_transform(data["description"])

    # Topic Modeling
    lda = LatentDirichletAllocation(n_components=n_clusters, random_state=42)
    lda_matrix = lda.fit_transform(tfidf_matrix)

    # Sentiment Analysis
    data["sentiment"] = data["description"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    # Combine features
    combined_features = np.hstack(
        (data[["amount"]].values, lda_matrix, data[["sentiment"]].values.reshape(-1, 1))
    )

    # Clustering
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    data["cluster"] = model.fit_predict(combined_features)

    return data
