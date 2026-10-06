# Databricks notebook source
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

df = spark.table("default.titanic_encoded").toPandas()
print(df.shape)

# COMMAND ----------

# DBTITLE 1,Cell 3
X = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
y = df["Survived"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC #10 Trees:

# COMMAND ----------

rf_model_10 = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

# COMMAND ----------

rf_model_10.fit(X_train, y_train)

# COMMAND ----------

y_pred_rf10 = rf_model_10.predict(X_test)
print(y_pred_rf10)

# COMMAND ----------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

rf_accuracy = accuracy_score(y_test, y_pred_rf10)
rf_precision = precision_score(y_test, y_pred_rf10)
rf_recall = recall_score(y_test, y_pred_rf10)
rf_f1 = f1_score(y_test, y_pred_rf10)
cm_rf = confusion_matrix(y_test, y_pred_rf10)

print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-score:", rf_f1)
print("Confusion Matrix:",cm_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC #100 Trees :

# COMMAND ----------

rf_model_100 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# COMMAND ----------

rf_model_100.fit(X_train, y_train)

# COMMAND ----------

y_pred_rf100 = rf_model_100.predict(X_test)
print(y_pred_rf100)

# COMMAND ----------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

rf_accuracy = accuracy_score(y_test, y_pred_rf100)
rf_precision = precision_score(y_test, y_pred_rf100)
rf_recall = recall_score(y_test, y_pred_rf100)
rf_f1 = f1_score(y_test, y_pred_rf100)
cm_rf = confusion_matrix(y_test, y_pred_rf100)

print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-score:", rf_f1)
print("Confusion Matrix:",cm_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC #200 Trees :

# COMMAND ----------

rf_model200 = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# COMMAND ----------

rf_model200.fit(X_train, y_train)

# COMMAND ----------

y_pred_rf200 = rf_model200.predict(X_test)
print(y_pred_rf200)

# COMMAND ----------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

rf_accuracy = accuracy_score(y_test, y_pred_rf200)
rf_precision = precision_score(y_test, y_pred_rf200)
rf_recall = recall_score(y_test, y_pred_rf200)
rf_f1 = f1_score(y_test, y_pred_rf200)
cm_rf = confusion_matrix(y_test, y_pred_rf200)

print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-score:", rf_f1)
print("Confusion Matrix:",cm_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC #MAX DEPTH :

# COMMAND ----------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    max_depth=5,
    random_state=42
)

# COMMAND ----------

rf_model.fit(X_train, y_train)

# COMMAND ----------

y_pred_rf = rf_model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import accuracy_score

rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)

# COMMAND ----------

from sklearn.metrics import precision_score, recall_score, f1_score

rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-score:", rf_f1)


# COMMAND ----------

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

# COMMAND ----------

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_rf))

# COMMAND ----------

# MAGIC %md
# MAGIC #FEATURE IMPORTANCE :

# COMMAND ----------

import pandas as pd
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)

# COMMAND ----------

# DBTITLE 1,Cell 30
import matplotlib.pyplot as plt

top_features = feature_importance.head(10)

top_features.sort_values().plot(kind="barh", figsize=(8, 5))

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importances")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #MODEL COMPARISON : 

# COMMAND ----------

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.8268,
        0.8045,
        0.8268
    ],
    "Precision": [
        0.7867,
        0.7671,
        0.8116
    ],
    "Recall": [
        0.7973,
        0.7568,
        0.7568
    ],
    "F1-score": [
        0.7919,
        0.7619,
        0.7832
    ]
})

comparison

# COMMAND ----------

comparison.plot(
    x="Model",
    y="Accuracy",
    kind="bar",
    figsize=(8, 5),
    legend=False
)

plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

# COMMAND ----------

comparison.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Selection
# MAGIC
# MAGIC Three classification models were evaluated on the same test set:
# MAGIC
# MAGIC | Model | Accuracy | Precision | Recall | F1-score |
# MAGIC |---|---:|---:|---:|---:|
# MAGIC | Logistic Regression | 82.68% | 78.67% | 79.73% | 79.19% |
# MAGIC | Decision Tree | 80.45% | 76.71% | 75.68% | 76.19% |
# MAGIC | Random Forest | 82.68% | 81.16% | 75.68% | 78.32% |
# MAGIC
# MAGIC Logistic Regression was selected as the final model for this project because it achieved the same accuracy as Random Forest while providing higher recall and F1-score on the test set.
# MAGIC
# MAGIC This selection is based on the results obtained from this Titanic test split and does not imply that Logistic Regression is universally better than Random Forest.