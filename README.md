# Indonesian Hoax News Detection: Experiments with MLflow

Experiment repository for Dicoding's *Membangun Sistem Machine Learning* (MLOps) class. It classifies Indonesian news articles as **hoax** or **valid**, from raw data through automated preprocessing to tracked model training.

The CI/CD part of the same project lives in **[ffyzhra/Workflow-CI](https://github.com/ffyzhra/Workflow-CI)**.

## Workflow

1. **Exploration & preprocessing** ([`preprocessing/Eksperimen_Fayza.ipynb`](./preprocessing/Eksperimen_Fayza.ipynb)): cleaning 500 labelled articles (250 hoax, 250 valid) from `dataset_raw/dataset_hoax.csv`: lowercasing, removing URLs, numbers and punctuation, tokenising and removing Indonesian stopwords (NLTK).
2. **Automated preprocessing** ([`preprocessing/automate_fayza.py`](./preprocessing/automate_fayza.py)): the same steps as a reusable script that writes `dataset_preprocessing.csv`.
3. **Baseline model** ([`Membangun_model/modelling.py`](./Membangun_model/modelling.py)): TF-IDF (5,000 features) + Logistic Regression, logged with `mlflow.sklearn.autolog()`.
4. **Tuned model** ([`Membangun_model/modelling_tuning.py`](./Membangun_model/modelling_tuning.py)): `GridSearchCV` over the regularisation strength `C`, with manual logging of parameters and accuracy, precision, recall and F1.

Screenshots of the MLflow dashboard and saved artifacts are in `Membangun_model/`.

## Running it

```bash
cd Membangun_model
pip install -r requirements.txt
python modelling.py           # baseline
python modelling_tuning.py    # tuned
mlflow ui                     # open http://127.0.0.1:5000
```

## Tech stack

Python · pandas · scikit-learn · NLTK · MLflow

## Author

**Fayza Azzahra** · [GitHub](https://github.com/ffyzhra) · [LinkedIn](https://www.linkedin.com/in/ffyzhra)
