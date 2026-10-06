# Databricks notebook source
df = spark.table("default.titanic_encoded").toPandas()
df.shape

# COMMAND ----------

X = df.drop(columns= ["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
y = df["Survived"]

X.shape, y.shape

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model =LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# COMMAND ----------

X.columns.tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC #FIRST PASSENGER :#

# COMMAND ----------

new_passenger = {
    "Pclass" : 1,
    "Sex" : "female",
    "Age" : 25,
    "SibSp" : 0,
    "Parch" : 0,
    "Fare" : 80,
    "Embarked" : "C",
    "Title" : "Miss",
    "Cabin_Deck" : "Unknown"
}

new_passenger

# COMMAND ----------

new_passenger["FamilySize"] = (
    new_passenger["SibSp"] +
    new_passenger["Parch"] +
    1
)

new_passenger

# COMMAND ----------

new_passenger["IsAlone"] = int(new_passenger["FamilySize"] == 1)
new_passenger

# COMMAND ----------

new_passenger["Age"] = 25
age = new_passenger["Age"]

if age <= 12:
    new_passenger["AgeGroup"] = "Child"
elif age <=17:
    new_passenger["AgeGroup"] = "Teen"
elif age <50:
    new_passenger["AgeGroup"] = "Adult"
else:
    new_passenger["AgeGroup"] = "Senior"
new_passenger
     

# COMMAND ----------

family_size = new_passenger["FamilySize"]

if family_size == 1:
    new_passenger["FamilySizeGroup"] = "Alone"
elif family_size <= 4:
    new_passenger["FamilySizeGroup"] = "Small"
elif family_size <= 6:
    new_passenger["FamilySizeGroup"] = "Medium"
else:
    new_passenger["FamilySizeGroup"] = "Large"

new_passenger

# COMMAND ----------

new_passenger["Sex"] = {"male" : 0, "female" : 1}[new_passenger["Sex"]]

new_passenger

# COMMAND ----------

new_passenger["Embarked_C"] = int(new_passenger["Embarked"] == "C")
new_passenger["Embarked_Q"] = int(new_passenger["Embarked"] == "Q")
new_passenger["Embarked_S"] = int(new_passenger["Embarked"] == "S")

new_passenger

# COMMAND ----------

new_passenger["Title_Master"] = int(new_passenger["Title"] == "Master")
new_passenger["Title_Miss"] = int(new_passenger["Title"] == "Miss")
new_passenger["Title_Mr"] = int(new_passenger["Title"] == "Mr")
new_passenger["Title_Mrs"] = int(new_passenger["Title"] == "Mrs")
new_passenger["Title_Other"] = int(
    new_passenger["Title"] == "Other"
)

new_passenger

# COMMAND ----------

new_passenger["AgeGroup_Child"] = int(new_passenger["AgeGroup"] == "Child")
new_passenger["AgeGroup_Teen"] = int(new_passenger["AgeGroup"] == "Teen")
new_passenger["AgeGroup_Adult"] = int(new_passenger["AgeGroup"] == "Adult")
new_passenger["AgeGroup_Senior"] = int(new_passenger["AgeGroup"] == "Senior")

new_passenger

# COMMAND ----------

new_passenger["FamilySizeGroup_Alone"] = int(
    new_passenger["FamilySizeGroup"] == "Alone"
)

new_passenger["FamilySizeGroup_Small"] = int(
    new_passenger["FamilySizeGroup"] == "Small"
)

new_passenger["FamilySizeGroup_Medium"] = int(
    new_passenger["FamilySizeGroup"] == "Medium"
)

new_passenger["FamilySizeGroup_Large"] = int(
    new_passenger["FamilySizeGroup"] == "Large"
)

new_passenger



# COMMAND ----------

cabin_decks = ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"]

for deck in cabin_decks:
    new_passenger[f"Cabin_Deck_{deck}"] = int(
        new_passenger["Cabin_Deck"] == deck
    )

new_passenger

# COMMAND ----------

import pandas as pd
new_passenger_df = pd.DataFrame([new_passenger])
new_passenger_df

# COMMAND ----------

new_passenger_df = new_passenger_df[X.columns]
new_passenger_df

# COMMAND ----------

new_passenger_df.shape

# COMMAND ----------

new_passenger_df.columns.tolist()

# COMMAND ----------

prediction = model.predict(new_passenger_df)
prediction

# COMMAND ----------

probability = model.predict_proba(new_passenger_df)
probability

# COMMAND ----------

# MAGIC %md
# MAGIC #SECOND PASSENGER :#

# COMMAND ----------

second_passenger = {
    "Pclass" : 3,
    "Sex" : "male",
    "Age" : 30,
    "SibSp" : 1,
    "Parch" : 0,
    "Fare" : 15,
    "Embarked" : 5,
    "Title" : "Mr",
    "Cabin_Deck" : "Unknown"
}

# COMMAND ----------

second_passenger["FamilySize"] = (
    second_passenger["SibSp"] +
    second_passenger["Parch"] + 1
)
second_passenger

# COMMAND ----------

second_passenger["IsAlone"] = int(second_passenger["FamilySize"]==1)
second_passenger

# COMMAND ----------

age = second_passenger["Age"]

if age <= 12:
    second_passenger["AgeGroup"] = "Child"
elif age <=17:
    second_passenger["AgeGroup"] = "Teen"
elif age <=59:
    second_passenger["AgeGroup"] = "Adult"
else:
    second_passenger["AgeGroup"] = "Senior"

second_passenger

# COMMAND ----------

familysize = second_passenger["FamilySize"]

if familysize == 1:
    second_passenger["FamilySizeGroup"] = "Alone"
elif familysize <=4:
    second_passenger["FamilySizeGroup"] = "Small"
elif familysize <=6:
    second_passenger["FamilySizeGroup"] = "Medium"
else:
    second_passenger["FamilySizeGroup"] = "Large"

second_passenger

# COMMAND ----------

second_passenger["Sex"] = {"male" : 0, "female" : 1}[second_passenger["Sex"]]

second_passenger

# COMMAND ----------

second_passenger["Embarked_C"] = int(second_passenger["Embarked"] =="C")
second_passenger["Embarked_Q"] = int(second_passenger["Embarked"] =="Q")
second_passenger["Embarked_S"] = int(second_passenger["Embarked"] =="S")

second_passenger

# COMMAND ----------

second_passenger["Title_Master"] = int(second_passenger["Title"]== "Master")
second_passenger["Title_Miss"] = int(second_passenger["Title"]== "Miss")
second_passenger["Title_Mr"] = int(second_passenger["Title"]== "Mr")
second_passenger["Title_Mrs"] = int(second_passenger["Title"]== "Mrs")
second_passenger["Title_Other"] = int(second_passenger["Title"]== "Other")

second_passenger

# COMMAND ----------

second_passenger["AgeGroup_Child"] = int(second_passenger["AgeGroup"]== "Child")
second_passenger["AgeGroup_Teen"] = int(second_passenger["AgeGroup"]== "Teen")
second_passenger["AgeGroup_Adult"] = int(second_passenger["AgeGroup"]== "Adult")
second_passenger["AgeGroup_Senior"]= int(second_passenger["AgeGroup"]== "Senior")

second_passenger




# COMMAND ----------

second_passenger["FamilySizeGroup_Alone"] = int(second_passenger["FamilySizeGroup"] == "Alone")
second_passenger["FamilySizeGroup_Small"] = int(second_passenger["FamilySizeGroup"] == "Small")
second_passenger["FamilySizeGroup_Medium"] = int(second_passenger["FamilySizeGroup"] == "Medium")
second_passenger["FamilySizeGroup_Large"] = int(second_passenger["FamilySizeGroup"] == "Large")

second_passenger


# COMMAND ----------

cabin_decks = ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"]
for deck in cabin_decks:
    second_passenger[f"Cabin_Deck_{deck}"] = int(
        second_passenger["Cabin_Deck"]== deck
    )

second_passenger

# COMMAND ----------

second_passenger_df = pd.DataFrame([second_passenger])
second_passenger_df


# COMMAND ----------

second_passenger_df = second_passenger_df[X.columns]
second_passenger_df.shape

# COMMAND ----------

second_prediction = model.predict(second_passenger_df)
second_prediction

# COMMAND ----------

second_probability = model.predict_proba(second_passenger_df)
second_probability

# COMMAND ----------

X.columns.tolist()

# COMMAND ----------

X_clean = X.drop(columns=["FamilySizeGroup_Alone"])
X_clean.shape

# COMMAND ----------

X_clean_train, X_clean_test, y_clean_train, y_clean_test = train_test_split(
    X_clean,
    y,
    test_size = 0.20,
    random_state =    42                                                         
)

# COMMAND ----------

clean_model = LogisticRegression(max_iter=1000)
clean_model.fit(X_clean_train, y_clean_train)

# COMMAND ----------

clean_y_pred = clean_model.predict(X_clean_test)
clean_y_pred[:10]

# COMMAND ----------

from sklearn.metrics import accuracy_score

clean_accuracy = accuracy_score(y_clean_test, clean_y_pred)

clean_accuracy

# COMMAND ----------

from sklearn.metrics import precision_score

clean_precision = precision_score(y_clean_test, clean_y_pred)

clean_precision

# COMMAND ----------

from sklearn.metrics import recall_score
clean_recall = recall_score(y_clean_test, clean_y_pred)

clean_recall

# COMMAND ----------

from sklearn.metrics import f1_score
clean_f1 = f1_score(y_clean_test, clean_y_pred)

clean_f1

# COMMAND ----------

from sklearn.metrics import confusion_matrix
clean_cm = confusion_matrix(y_clean_test, clean_y_pred)

clean_cm