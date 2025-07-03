from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__,template_folder='./templates',static_folder='./static')

# Add MySQL configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Harshil@localhost/fake_news_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define News model
class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

loaded_model = pickle.load(open("model.pkl", 'rb'))
vector = pickle.load(open("vector.pkl", 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))
corpus = []

def fake_news_det(news):
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
     
    return prediction

def preprocess_text(text):
    # Apply the same preprocessing as in fake_news_det
    review = re.sub(r'[^a-zA-Z\s]', '', text)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review:
        if y not in stpwrds:
            corpus.append(lemmatizer.lemmatize(y))
    return ' '.join(corpus)

def retrain_model():
    try:
        # Get data from database from the last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_entries = News.query.filter(News.created_at >= thirty_days_ago).all()
        
        if len(recent_entries) < 50:  # Minimum threshold for retraining
            return False, "Not enough new data for retraining"
        
        # Prepare data for training
        texts = [entry.content for entry in recent_entries]
        labels = [1 if entry.prediction else 0 for entry in recent_entries]
        
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Create new vectorizer and transform texts
        new_vectorizer = TfidfVectorizer(max_features=5000)
        X = new_vectorizer.fit_transform(processed_texts)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        # Initialize and train new model
        new_model = PassiveAggressiveClassifier(max_iter=50)
        new_model.fit(X_train, y_train)
        
        # Evaluate new model
        y_pred = new_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy >= 0.85:  # Only save if accuracy is good enough
            # Save new model and vectorizer
            with open('model.pkl', 'wb') as f:
                pickle.dump(new_model, f)
            with open('vector.pkl', 'wb') as f:
                pickle.dump(new_vectorizer, f)
            
            # Update global variables
            global loaded_model, vector
            loaded_model = new_model
            vector = new_vectorizer
            
            return True, f"Model retrained successfully with accuracy: {accuracy:.2f}"
        else:
            return False, f"New model accuracy ({accuracy:.2f}) below threshold"
            
    except Exception as e:
        return False, f"Error during retraining: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/history')
def history():
    news_entries = News.query.order_by(News.created_at.desc()).all()
    return render_template('history.html', entries=news_entries)

@app.route('/retrain', methods=['POST'])
def retrain():
    success, message = retrain_model()
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)
        
        # Store in database
        news_entry = News(
            content=message,
            prediction=(pred[0] == 1)  # True if fake news, False if real
        )
        db.session.add(news_entry)
        db.session.commit()
        
        # Check if we should retrain (e.g., every 100 entries)
        if News.query.count() % 100 == 0:
            retrain_model()
        
        def predi(pred):
            if pred[0] == 1:
                res="Prediction of the News :  Looking Fake News📰"
            else:
                res="Prediction of the News : Looking Real News📰 "
            return res
        
        result=predi(pred)
        return render_template("prediction.html",  prediction_text="{}".format(result))
    else:
        return render_template('prediction.html', prediction="Something went wrong")

if __name__ == '__main__':
    print("Starting Flask app...")  # Add this line
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True, port=5001)