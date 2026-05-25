# Simple Streamlit app to compare ML models on a CSV file.
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from catboost import CatBoostClassifier


# Page title and short description
st.title("ML Model Comparison")
st.write(
    "Upload a CSV, pick the target column, choose one or more models, "
    "and compare their performance."
)


# Step 1: upload a CSV file
st.subheader("1. Upload CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Stop here if no file has been uploaded yet
if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()


# Read the CSV into a DataFrame
df = pd.read_csv(uploaded_file)


# Step 2: show a small preview of the data
st.subheader("2. Data preview")
st.dataframe(df.head())
st.write("Shape:", df.shape)


# Step 3: select the target column
st.subheader("3. Select target column")
target_column = st.selectbox("Target column", df.columns)


# Step 4: select one or more models
st.subheader("4. Select models")
model_options = ["SVM", "Random Forest", "KNN", "Neural Network", "CatBoost"]
selected_models = st.multiselect("Models", model_options, default=model_options)


# Step 5: training button
st.subheader("5. Train and evaluate")
train_button = st.button("Train and evaluate")

# Only run the training when the user clicks the button
if not train_button:
    st.stop()

# Make sure the user picked at least one model
if len(selected_models) == 0:
    st.warning("Please select at least one model.")
    st.stop()


# Simple preprocessing
# Drop rows with missing values to keep things simple
df = df.dropna()

# Separate features (X) and target (y)
y = df[target_column]
X = df.drop(columns=[target_column])

# Turn text labels (e.g. Yes/No) into numbers 0, 1, 2, ...
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# One-hot encode non-numeric columns in X
X = pd.get_dummies(X, drop_first=True)

# Check that the target has at least 2 classes
classes = np.unique(y)
if len(classes) < 2:
    st.error("The target column must have at least 2 classes.")
    st.stop()

# Train/test split (stratified so class balance is kept)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (helps SVM, KNN and Neural Network)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Dictionary that maps a model name to a fresh estimator
all_models = {
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(max_iter=500, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
}


# Train each selected model and store its metrics in a list
results = []

# Detect if the problem is binary or multiclass
is_binary = len(classes) == 2

# For multiclass average precision we need a one-hot version of y_test
if not is_binary:
    lb = LabelBinarizer()
    lb.fit(y_train)
    y_test_bin = lb.transform(y_test)

try:
    for name in selected_models:
        model = all_models[name]

        # Train the model
        model.fit(X_train, y_train)

        # Predict labels and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Recall, Precision and F1 (weighted handles binary and multiclass)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC-AUC and Average Precision (PR-AUC):
        # binary uses the positive class column,
        # multiclass uses one-vs-rest with weighted average
        if is_binary:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            pr_auc = average_precision_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted"
            )
            pr_auc = average_precision_score(
                y_test_bin, y_proba, average="weighted"
            )

        # Save the metrics for this model
        results.append(
            {
                "Model": name,
                "Recall": recall,
                "Precision": precision,
                "F1-score": f1,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc,
            }
        )
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()


# Build a DataFrame with the results
results_df = pd.DataFrame(results)


# Show the results table
st.subheader("Results")
st.dataframe(
    results_df.style.format(
        {
            "Recall": "{:.3f}",
            "Precision": "{:.3f}",
            "F1-score": "{:.3f}",
            "ROC-AUC": "{:.3f}",
            "PR-AUC": "{:.3f}",
        }
    )
)


# Show a bar chart comparing the metrics for each model
st.subheader("Model comparison")
chart_df = results_df.set_index("Model")
st.bar_chart(chart_df)
