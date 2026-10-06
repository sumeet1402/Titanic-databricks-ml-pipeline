# Databricks notebook source
from sklearn.model_selection import train_test_split

# COMMAND ----------

df = spark.table("default.titanic_encoded").toPandas()

# COMMAND ----------

X = df.drop(columns= ["PassengerId", "Name", "Ticket", "Cabin", "Survived"])

y = df["Survived"]

# COMMAND ----------

X.shape , y.shape

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size= 0.20,
    random_state = 42
)

# COMMAND ----------

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# COMMAND ----------

