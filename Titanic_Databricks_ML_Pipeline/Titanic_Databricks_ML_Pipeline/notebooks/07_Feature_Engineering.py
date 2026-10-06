# Databricks notebook source
import numpy as np
import pandas as pd


# COMMAND ----------

df= spark.table("default.titanic_clean").toPandas()

# COMMAND ----------

df["PassengerId"].duplicated().sum()

# COMMAND ----------

df["PassengerId"].nunique()

# COMMAND ----------

df = df.drop_duplicates(subset="PassengerId", keep = "first")

# COMMAND ----------

df.shape

# COMMAND ----------

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# COMMAND ----------

df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# COMMAND ----------

df[["SibSp","Parch","FamilySize", "IsAlone"]].head(10)

# COMMAND ----------

df["Name"].head(10)

# COMMAND ----------

df["Title"] = df["Name"].str.extract(r',\s*([^.]*)\.')

# COMMAND ----------

df[["Name", "Title"]].head(10)

# COMMAND ----------

df["Title"].value_counts()

# COMMAND ----------

common_titles = ["Mr", "Miss", "Mrs", "Master"]

# COMMAND ----------

df["Title"] = df["Title"].where(
    df["Title"].isin(common_titles),"Other"
)

# COMMAND ----------

df["Age"].describe()

# COMMAND ----------

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins = [0,12,17,59,100],
    labels=  ["Child", "Teen", "Adult", "Senior"]
)

# COMMAND ----------

df[["Age", "AgeGroup"]].head()

# COMMAND ----------

df["AgeGroup"].value_counts()

# COMMAND ----------

age_median = df["Age"].median()
age_median

# COMMAND ----------

df["Age"] = df["Age"].fillna(age_median)

# COMMAND ----------

df["AgeGroup"].value_counts()

# COMMAND ----------

df["FamilySizeGroup"] = pd.cut(
    df["FamilySize"],
    bins = [0,1,4,6, float("inf")],
    labels = ["Alone","Small","Medium","Large"]
)

# COMMAND ----------

df[["FamilySize", "FamilySizeGroup"]].head(10)

# COMMAND ----------

df["FamilySizeGroup"].value_counts()

# COMMAND ----------

df["Fare"].describe()

# COMMAND ----------

df[["Sex", "Embarked", "Title", "AgeGroup", "FamilySizeGroup"]].head(10)

# COMMAND ----------

df.select_dtypes(include="object").columns

# COMMAND ----------

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 17, 59, 100],
    labels=["Child", "Teen", "Adult", "Senior"]
)

# COMMAND ----------

df["Sex"] = df["Sex"].map({
    "male" : 0,
    "female" : 1
})

# COMMAND ----------

embarded_mode = df["Embarked"].mode()[0]
embarded_mode

# COMMAND ----------

df["Embarked"] = df["Embarked"].fillna(embarded_mode)

# COMMAND ----------

df["Title"].value_counts()

# COMMAND ----------

df = pd.get_dummies(df, columns=["Embarked"], dtype=int)

# COMMAND ----------

df[["Embarked_C", "Embarked_Q", "Embarked_S"]].head(10)

# COMMAND ----------

df = pd.get_dummies(df, columns = ["Title"], dtype = int)

# COMMAND ----------

df[[
    "Title_Master",
    "Title_Miss",
    "Title_Mr",
    "Title_Mrs",
    "Title_Other"
]].head(10)

# COMMAND ----------

df = pd.get_dummies(df, columns = ["AgeGroup"], dtype = int)

# COMMAND ----------

df[[
    "AgeGroup_Child",
    "AgeGroup_Teen",
    "AgeGroup_Adult",
    "AgeGroup_Senior"
]].head(10)

# COMMAND ----------

df = pd.get_dummies(df, columns = ["FamilySizeGroup"], dtype = int)


# COMMAND ----------

df[[
    "FamilySizeGroup_Alone",
    "FamilySizeGroup_Small",
    "FamilySizeGroup_Medium",
    "FamilySizeGroup_Large"
]].head(10)

# COMMAND ----------

df["Cabin_Deck"] = df["Cabin"].str[0]

# COMMAND ----------

df["Cabin_Deck"] = df["Cabin_Deck"].fillna("Unknown")

# COMMAND ----------

df["Cabin_Deck"].value_counts()

# COMMAND ----------

df = pd.get_dummies(df, columns= ["Cabin_Deck"], dtype = int)

# COMMAND ----------

df[[
    "Cabin_Deck_A",
    "Cabin_Deck_B",
    "Cabin_Deck_C",
    "Cabin_Deck_D",
    "Cabin_Deck_E",
    "Cabin_Deck_F",
    "Cabin_Deck_G",
    "Cabin_Deck_T",
    "Cabin_Deck_Unknown",
]].head(10)

# COMMAND ----------

X = df.drop(columns = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
y = df["Survived"]

# COMMAND ----------

X.shape, y.shape

# COMMAND ----------

spark_df = spark.createDataFrame(df)

# COMMAND ----------

spark_df.write.mode("overwrite").saveAsTable("default.titanic_encoded")

# COMMAND ----------

spark_df.count(), len(spark_df.columns)