# Databricks notebook source
df = spark.table("default.titanic_encoded").toPandas()

# COMMAND ----------

df.shape

# COMMAND ----------

X = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
y = df["Survived"]

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# COMMAND ----------

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

model = LogisticRegression(max_iter=1000)

# COMMAND ----------

model.fit(X_train, y_train)

# COMMAND ----------

import pandas as pd

coefficients = pd.DataFrame({
    "Feature" : X_train.columns,
    "Coefficient" : model.coef_[0]
    })

coefficients

# COMMAND ----------

model.predict(X_test)

# COMMAND ----------

model.predict_proba(X_test)[:5]