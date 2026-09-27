"""
SMS Spam Classifier - Model Training Script
Second Year Project
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

def main():
    print("1. Loading dataset...")
    df = pd.read_csv('sms_spam_collection (1).csv')
    print(f"   Loaded {len(df)} rows.")

    print("\n2. Cleaning data...")
    df = df.drop_duplicates().reset_index(drop=True)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    print(f"   After removing duplicates: {len(df)} rows ({df['target'].sum()} spam, {len(df) - df['target'].sum()} ham).")

    print("\n3. Splitting into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
    )

    print("\n4. Training Multinomial Naive Bayes model...")
    pipeline_nb = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
        ('nb', MultinomialNB())
    ])
    pipeline_nb.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline_nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    print("\n5. Fitting pipeline on full clean dataset & saving artifact...")
    pipeline_nb.fit(df['message'], df['target'])
    joblib.dump(pipeline_nb, 'sms_spam_model.joblib')
    print("   Saved pipeline to 'sms_spam_model.joblib' successfully!")

if __name__ == '__main__':
    main()
