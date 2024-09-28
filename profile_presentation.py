import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['default'])
    y = data['default']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def expected_loss(model, borrower_data, loan_amount, recovery_rate=0.10):
    pd_value = model.predict_proba(borrower_data)[:, 1]
    return pd_value * loan_amount * (1 - recovery_rate)

def calculate_total_loss(model, X_test, min_loan=50000, max_loan=200000):
    loan_amounts = np.random.randint(min_loan, max_loan, size=len(X_test))
    losses = expected_loss(model, X_test, loan_amounts)
    return np.sum(losses)

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data('loan_data.csv')
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Example for a single borrower
    borrower_data = X_test.iloc[[0]]
    loan_amount = 100000
    loss = expected_loss(model, borrower_data, loan_amount)[0]
    print(f'Expected Loss for the example borrower: ${loss:.2f}')

    # Total expected loss for the test set
    total_loss = calculate_total_loss(model, X_test)
    print(f'Total Expected Loss for the test set: ${total_loss:.2f}')

if __name__ == "__main__":
    main()