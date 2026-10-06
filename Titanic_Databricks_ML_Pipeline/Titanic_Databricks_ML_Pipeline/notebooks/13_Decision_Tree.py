# Databricks notebook source
from sklearn.tree import DecisionTreeClassifier

# COMMAND ----------

df = spark.table("default.titanic_encoded").toPandas()
print(df.shape)

# COMMAND ----------

X = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
y = df["Survived"]


# COMMAND ----------

print(X.shape)
print(y.shape)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state = 42
)

# COMMAND ----------

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# COMMAND ----------

tree_model = DecisionTreeClassifier(random_state= 24)

# COMMAND ----------

tree_model.fit(X_train, y_train)

# COMMAND ----------

y_pred_tree = tree_model.predict(X_test)

# COMMAND ----------

print(y_pred_tree[:10])
print(y_test[:10])

# COMMAND ----------

from sklearn.metrics import accuracy_score

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print(tree_accuracy)

# COMMAND ----------

from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)
print(cm_tree)

# COMMAND ----------

from sklearn.metrics import precision_score

tree_precision = precision_score(y_test, y_pred_tree)

print(tree_precision)

# COMMAND ----------

from sklearn.metrics import recall_score

tree_recall = recall_score(y_test, y_pred_tree)

print(tree_recall)

# COMMAND ----------

from sklearn.metrics import f1_score

tree_f1 = f1_score(y_test, y_pred_tree)

print(tree_f1)

# COMMAND ----------

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_tree))

# COMMAND ----------

