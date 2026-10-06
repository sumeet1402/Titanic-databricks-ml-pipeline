# Databricks notebook source
df = spark.table("default.titanic_encoded").toPandas()
df.shape

# COMMAND ----------

X = df.drop(columns= ["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
y = df["Survived"]

X.shape, y.shape

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.20,
    random_state= 42
)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# COMMAND ----------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# COMMAND ----------

y_pred  = model.predict(X_test)

y_pred

# COMMAND ----------

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy

# COMMAND ----------

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# COMMAND ----------

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
precision

# COMMAND ----------

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
recall

# COMMAND ----------

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
f1

# COMMAND ----------

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



# COMMAND ----------

