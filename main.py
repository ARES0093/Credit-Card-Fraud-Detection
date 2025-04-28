import pandas as pd

train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')

print("Train set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print(train_df['is_fraud'].value_counts(normalize=True))
print(train_df.info())
print(train_df.describe())

from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(df, fit_scaler=True, scaler=None):
    df = df.copy()

    # Drop columns not useful for modeling
    df = df.drop(['trans_date_trans_time', 'cc_num', 'first', 'last', 'dob', 'unix_time', 'merchant', 'street', 'city', 'state', 'zip'], axis=1, errors='ignore')

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Scale numerical features
    num_cols = df.select_dtypes(include=['float64', 'int64']).drop('is_fraud', axis=1, errors='ignore').columns
    if fit_scaler:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df, scaler
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        return df

from imblearn.over_sampling import SMOTE

def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)

from xgboost import XGBClassifier

def train_model(X_train, y_train):
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label='XGBoost')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
import shap

def explain_model(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)
import joblib

def save_model(model, scaler):
    joblib.dump(model, 'fraud_detection_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
if __name__ == "__main__":
    # Preprocess training data
    train_df_clean, scaler = preprocess(train_df, fit_scaler=True)
    X_train = train_df_clean.drop('is_fraud', axis=1)
    y_train = train_df_clean['is_fraud']

    # Apply SMOTE
    X_resampled, y_resampled = apply_smote(X_train, y_train)

    # Train the model
    model = train_model(X_resampled, y_resampled)

    # Preprocess test data using same scaler
    test_df_clean = preprocess(test_df, fit_scaler=False, scaler=scaler)
    X_test = test_df_clean.drop('is_fraud', axis=1)
    y_test = test_df_clean['is_fraud']

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Explainability
    explain_model(model, X_test.sample(100))

    # Save model
    save_model(model, scaler)
