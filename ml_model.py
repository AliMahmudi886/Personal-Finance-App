import numpy as np
import pandas as pd
import sqlite3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from flask import Flask
from tensorflow.keras.layers import Input

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

app = Flask(__name__)


def fetch_expense_data(user_id):
    conn = sqlite3.connect('personal_finance.db')
    query = "SELECT date, amount, description FROM expenses WHERE user_id = ?"
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()

    # Replace None with empty string in 'description' column
    df['description'] = df['description'].fillna('')

    # Ensure 'description' column is of type string
    df['description'] = df['description'].astype(str)

    return df




def train_lstm_model(user_id):
    data = fetch_expense_data(user_id)
    if data.empty or data.shape[0] < 2:
        return None, None

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data.resample('ME').sum()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['amount']])

    X, y = [], []
    for i in range(len(scaled_data) - 1):
        X.append(scaled_data[i])
        y.append(scaled_data[i + 1])
    X, y = np.array(X), np.array(y)

    if len(X) < 1:
        return None, None

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=1, verbose=2)

    return model, scaler

def predict_next_month_lstm(user_id, model, scaler):
    if not model or not scaler:
        return None

    data = fetch_expense_data(user_id)
    if data.empty:
        return None

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data.resample('ME').sum()

    if data.shape[0] < 1:
        return None

    last_value = data['amount'].values[-1]
    scaled_last_value = scaler.transform([[last_value]])
    scaled_last_value = np.reshape(scaled_last_value, (1, 1, 1))

    prediction = model.predict(scaled_last_value)
    prediction = scaler.inverse_transform(prediction)
    next_month_prediction = prediction[0, 0]

    return next_month_prediction

def detect_anomalies(user_id):
    data = fetch_expense_data(user_id)
    if data.empty:
        return pd.DataFrame()

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data.resample('ME').sum()

    if data.shape[0] < 2:
        return pd.DataFrame()

    model = IsolationForest(contamination=0.1)
    data['anomaly'] = model.fit_predict(data[['amount']])
    anomalies = data[data['anomaly'] == -1].drop(columns=['anomaly'])

    return anomalies


def recommend_savings_plan(user_id):
    data = fetch_expense_data(user_id)
    if data.empty:
        return None

    if data.shape[0] < 2:
        return None

    model = NearestNeighbors(n_neighbors=1)
    model.fit(data[['amount']])

    last_amount = data['amount'].values[-1]
    recommendation = model.kneighbors([[last_amount]], return_distance=False)
    recommended_amount = data.iloc[recommendation[0][0]]['amount']

    return recommended_amount

def train_autoencoder(user_id):
    data = fetch_expense_data(user_id)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data.resample('M').sum()

    if data.shape[0] < 2:
        return None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    input_dim = scaled_data.shape[1]
    encoding_dim = input_dim // 2

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh")(input_layer)
    decoder = Dense(input_dim, activation="linear")(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=8, shuffle=True, verbose=2)

    return autoencoder, scaler

def detect_anomalies_autoencoder(user_id, model, scaler):
    data = fetch_expense_data(user_id)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data.resample('ME').sum()

    if data.shape[0] < 2:
        return None

    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)
    mse = np.mean(np.power(scaled_data - predictions, 2), axis=1)
    anomaly_threshold = np.percentile(mse, 95)
    data['anomaly'] = mse > anomaly_threshold

    return data[data['anomaly']]



def cluster_expenses(user_id, method='kmeans', n_clusters=3):
    data = fetch_expense_data(user_id)
    if data.empty or 'amount' not in data.columns or 'description' not in data.columns:
        return None

    # Check if there are non-empty descriptions
    if data['description'].str.strip().empty:
        print("No non-empty descriptions found.")
        return None

    # Filter out rows with empty descriptions
    data = data[data['description'].str.strip() != '']

    # Text processing for NLP
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['description'])

    # Topic Modeling
    lda = LatentDirichletAllocation(n_components=n_clusters, random_state=42)
    lda_matrix = lda.fit_transform(tfidf_matrix)

    # Sentiment Analysis
    data['sentiment'] = data['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Combine features
    combined_features = np.hstack((data[['amount']].values, lda_matrix, data[['sentiment']].values.reshape(-1, 1)))

    # Clustering
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    data['cluster'] = model.fit_predict(combined_features)

    return data
