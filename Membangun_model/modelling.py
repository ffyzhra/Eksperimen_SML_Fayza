import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv('preprocessing/dataset_preprocessing.csv')

# Feature dan target
X = df['text']
y = df['label']

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Mulai MLflow run
with mlflow.start_run():

    # Model
    model = LogisticRegression()

    # Training
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))