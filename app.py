import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_risk.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# --- Load ML Models ---
models = {}
scaler = None

def load_models():
    """Load all ML models into memory."""
    global scaler
    global models
    try:
        # Load scaler
        scaler_path = os.path.join('ml_models', 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Loaded feature scaler.")

        diseases = ['diabetes', 'hypertension', 'heart_disease']
        model_types = ['rf', 'xgb', 'svm']

        for d in diseases:
            models[d] = {}
            for t in model_types:
                path = os.path.join('ml_models', f'{t}_{d}.pkl')
                if os.path.exists(path):
                    models[d][t] = joblib.load(path)
                    print(f"Loaded {t.upper()} model for {d.replace('_', ' ').title()}")
                else:
                    print(f"Warning: Model {path} not found.")
    except Exception as e:
        print(f"Error loading models: {e}")


# Load models at startup
load_models()


# --- Routes ---

@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email address already exists', 'error')
            return redirect(url_for('signup'))

        username_exists = User.query.filter_by(username=username).first()
        if username_exists:
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        flash('Account created successfully!', 'success')
        return redirect(url_for('home'))

    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Parse form data
        data = {
            'age': float(request.form.get('age')),
            'gender': 1 if request.form.get('gender') == 'Male' else 0,
            'bmi': float(request.form.get('bmi')),
            'blood_pressure_systolic': float(request.form.get('blood_pressure_systolic')),
            'blood_pressure_diastolic': float(request.form.get('blood_pressure_diastolic')),
            'cholesterol': float(request.form.get('cholesterol')),
            'glucose': float(request.form.get('glucose')),
            
            # Categorical encoding MUST match train_models.py
            'smoking': {'Non-smoker': 0, 'Former smoker': 1, 'Current smoker': 2}.get(request.form.get('smoking'), 0),
            'alcohol': {'None': 0, 'Moderate': 1, 'Heavy': 2}.get(request.form.get('alcohol'), 0),
            'physical_activity': {'Low': 0, 'Moderate': 1, 'High': 2}.get(request.form.get('physical_activity'), 0),
            'family_history': {'None': 0, 'Diabetes': 1, 'Heart Disease': 2, 'Hypertension': 3}.get(request.form.get('family_history'), 0),
        }

        # Create DataFrame (1 row)
        feature_cols = [
            'age', 'gender', 'bmi', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'cholesterol', 'glucose',
            'smoking', 'alcohol', 'physical_activity', 'family_history'
        ]
        df_input = pd.DataFrame([data], columns=feature_cols)
        
        # Scale for SVM
        scaled_input = scaler.transform(df_input) if scaler else df_input

        results = {}
        diseases = ['diabetes', 'hypertension', 'heart_disease']
        
        # Predict for each disease with all 3 models
        for d in diseases:
            disease_results = {}
            if d not in models:
                continue

            # Random Forest
            if 'rf' in models[d]:
                rf_prob = models[d]['rf'].predict_proba(df_input)[0][1]
                disease_results['Random Forest'] = round(float(rf_prob) * 100, 1)

            # XGBoost
            if 'xgb' in models[d]:
                xgb_prob = models[d]['xgb'].predict_proba(df_input)[0][1]
                disease_results['XGBoost'] = round(float(xgb_prob) * 100, 1)

            # SVM
            if 'svm' in models[d]:
                svm_prob = models[d]['svm'].predict_proba(scaled_input)[0][1]
                disease_results['SVM'] = round(float(svm_prob) * 100, 1)

            # Calculate average risk
            if disease_results:
                avg_risk = sum(disease_results.values()) / len(disease_results)
                disease_results['Average'] = round(avg_risk, 1)
                
                # Determine risk level category
                if avg_risk < 20:
                    disease_results['Risk_Level'] = 'Low Risk'
                elif avg_risk < 50:
                    disease_results['Risk_Level'] = 'Medium Risk'
                else:
                    disease_results['Risk_Level'] = 'High Risk'

            results[d] = disease_results

        return render_template('results.html', results=results)

    except Exception as e:
        print(f"Prediction error: {e}")
        flash("Error processing prediction. Please check your inputs.", "error")
        return redirect(url_for('home'))

def init_db():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    # Initialize DB (run once on startup)
    if not os.path.exists('health_risk.db'):
        init_db()
        print("Initialized SQLite database.")
        
    app.run(debug=True, port=5000)
