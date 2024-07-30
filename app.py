from flask import Flask, redirect, url_for, session, request, render_template, flash, g
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
import os
from models import get_user_by_username, create_user, get_expenses_by_user_id, create_expense, get_goals_by_user_id, create_goal, get_user_by_id, update_user_profile, create_income, get_income_by_user_id, get_expenses_fortbl_by_user_id
import logging
import sqlite3
from ml_model import train_lstm_model,\
    predict_next_month_lstm, detect_anomalies, cluster_expenses, recommend_savings_plan,fetch_expense_data,detect_anomalies_autoencoder
from flask_bcrypt import Bcrypt
import pandas as pd
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = 'your_secret_key'
bcrypt = Bcrypt(app)
# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/profile_images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
# Configure OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='YOUR_GOOGLE_CLIENT_ID',
    client_secret='YOUR_GOOGLE_CLIENT_SECRET',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    client_kwargs={'scope': 'openid email profile'},
)
@app.route('/login/google')
def google_login():
    redirect_uri = url_for('google_authorized', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/authorized')
def google_authorized():
    token = google.authorize_access_token()
    user_info = google.parse_id_token(token)

    session['google_token'] = token
    session['user_id'] = user_info['sub']
    session['username'] = user_info['name']
    session['email'] = user_info['email']
    return redirect(url_for('home'))



# Function to run before every request
@app.before_request
def load_user():
    user_id = session.get('user_id')
    if user_id:
        g.user = get_user_by_id(user_id)
    else:
        g.user = None

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('personal_finance.db')
    except sqlite3.Error as e:
        print(e)
    return conn

@app.route('/')
def home():
    if 'user_id' in session:
        user_id = session['user_id']
        model, scaler = train_lstm_model(user_id)
        next_month_prediction = predict_next_month_lstm(user_id, model, scaler)
        return render_template('home.html', prediction=next_month_prediction)
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user_by_username(username)
        if user and bcrypt.check_password_hash(user[2], password):
            session['user_id'] = user[0]
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        if create_user(username, hashed_password):
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        else:
            flash('Registration failed. Username may already be taken.', 'danger')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/add_expense', methods=['GET', 'POST'])
def add_expense():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        category = request.form['category']
        amount = request.form['amount']
        date = request.form['date']
        create_expense(session['user_id'], category, amount, date)
        flash('Expense added!', 'success')
        return redirect(url_for('view_expenses'))
    return render_template('add_expense.html')

@app.route('/view_expenses')
def view_expenses():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    expenses = get_expenses_fortbl_by_user_id(session['user_id'])
    return render_template('view_expenses.html', expenses=expenses)

@app.route('/add_goal', methods=['GET', 'POST'])
def add_goal():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        goal = request.form['goal']
        target_amount = request.form['target_amount']
        current_amount = request.form['current_amount']
        create_goal(session['user_id'], goal, target_amount, current_amount)
        flash('Goal added!', 'success')
        return redirect(url_for('view_goals'))
    return render_template(' add_goal.html')

@app.route('/view_goals')
def view_goals():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    goals = get_goals_by_user_id(session['user_id'])
    return render_template(' view_goals.html', goals=goals)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = get_user_by_id(user_id)

    if request.method == 'POST':
        new_username = request.form['username']
        new_password = request.form.get('password', None)  # Get the new password if provided

        # If a new password is provided, hash it
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8') if new_password else None

        # Handle file upload
        profile_image = None
        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                profile_image = filename

        update_user_profile(user_id, new_username, hashed_password, profile_image)
        flash('Profile updated!', 'success')
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)

@app.route('/add_income', methods=['GET', 'POST'])
def add_income():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        source = request.form['source']
        amount = request.form['amount']
        date = request.form['date']
        create_income(session['user_id'], source, amount, date)
        flash('Income added!', 'success')
        return redirect(url_for('view_income'))

    return render_template('add_income.html')

@app.route('/view_income')
def view_income():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    income = get_income_by_user_id(session['user_id'])
    return render_template('view_income.html', income=income)


@app.route('/predict_expenses')
def predict_expenses():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    expenses = fetch_expense_data(user_id)
    if expenses.empty or expenses.shape[0] < 2:  # Check for sufficient historical data
        logging.warning(f"Not enough data available for user {user_id} to make a prediction.")
        return render_template('predict_expenses.html', error="Not enough data available to make a prediction.")

    model, scaler = train_lstm_model(user_id)
    if not model or not scaler:
        logging.error(f"Model training failed for user {user_id} due to insufficient data.")
        return render_template('predict_expenses.html', error="Not enough data to train the model.")

    next_month_prediction = predict_next_month_lstm(user_id, model, scaler)
    if next_month_prediction is None:
        logging.error(f"Prediction could not be made for user {user_id}")
        return render_template('predict_expenses.html', error="Not enough data to make a prediction.")

    next_month_prediction = float(next_month_prediction)

    expenses['date'] = pd.to_datetime(expenses['date'])
    expenses.set_index('date', inplace=True)
    expenses = expenses.resample('ME').sum()
    dates = expenses.index.strftime('%Y-%m').tolist()
    amounts = expenses['amount'].tolist()

    labels = dates + ['Next Month']
    actual_expenses = amounts + [None]
    predicted_expenses = [None] * len(amounts) + [next_month_prediction]

    return render_template('predict_expenses.html', prediction=next_month_prediction,
                           labels=labels, actual_expenses=actual_expenses, predicted_expenses=predicted_expenses)


@app.route('/view_anomalies')
def view_anomalies():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    anomalies = detect_anomalies(user_id)

    return render_template('view_anomalies.html', anomalies=anomalies)

@app.route('/detect_anomalies')
def detect_anomalies_route():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    anomalies = detect_anomalies(user_id)

    if anomalies is None:
        logging.error(f"Anomalies could not be detected for user {user_id}")
        return render_template('detect_anomalies.html', error="Not enough data to detect anomalies.")

    return render_template('detect_anomalies.html', anomalies=anomalies)

@app.route('/detect_anomalies_autoencoder')
def detect_anomalies_autoencoder_route():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    anomalies_autoencoder = detect_anomalies_autoencoder(user_id)

    if anomalies_autoencoder is None:
        logging.error(f"Anomalies could not be detected using autoencoder for user {user_id}")
        return render_template('detect_anomalies_autoencoder.html', error="Not enough data to detect anomalies using autoencoder.")

    return render_template('detect_anomalies_autoencoder.html', anomalies=anomalies_autoencoder)

@app.route('/expense_clusters')
def view_expense_clusters():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    try:
        clustered_data = cluster_expenses(user_id)
        clustered_data = clustered_data.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
    except ValueError as e:
        flash(str(e), 'danger')
        return redirect(url_for('home'))

    return render_template('view_clusters.html', clustered_data=clustered_data)

@app.route('/recommend_savings')
def recommend_savings():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    recommended_amount = recommend_savings_plan(user_id)

    return render_template('recommend_savings.html', recommended_amount=recommended_amount)


@app.route('/expenses/clusters')
def expense_clusters():
    if 'user_id' in session:
        user_id = session['user_id']
        method = request.args.get('method', 'kmeans')
        n_clusters = int(request.args.get('n_clusters', 3))

        clustered_data = cluster_expenses(user_id, method=method, n_clusters=n_clusters)
        if clustered_data is not None:
            clusters = clustered_data.groupby('cluster').apply(lambda x: x.to_dict(orient='records')).to_dict()
            return render_template('expense_clusters.html', clusters=clusters, method=method, n_clusters=n_clusters)

        flash('Not enough data for clustering')
        return redirect(url_for('view_expenses'))

    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
