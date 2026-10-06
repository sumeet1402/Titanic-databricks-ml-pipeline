# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Explore the structure, quality, and characteristics of the Titanic dataset before performing further analysis.
# MAGIC

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df = pd.read_csv("/Volumes/workspace/default/ml_data/Titanic-Dataset.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Preview the Dataset
# MAGIC
# MAGIC Preview the first few rows of the dataset to understand the type of information it contains.

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## 2. Check the Dataset Shape
# MAGIC
# MAGIC Check the number of rows and columns in the dataset.

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## 3. Display the Column Names
# MAGIC
# MAGIC Display the names of all columns in the dataset.

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## 4. Check the Data Types
# MAGIC
# MAGIC Check the data type of each column in the dataset.

# COMMAND ----------

df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect the DataFrame Information
# MAGIC
# MAGIC Inspect the number of entries, column names, non-null values, and data types.

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Check for Missing Values
# MAGIC
# MAGIC Check the number of missing values in each column.

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## 7. Check for Duplicate Rows
# MAGIC
# MAGIC Check whether the dataset contains any duplicate rows.

# COMMAND ----------

df.duplicated().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## 8. Count Unique Values
# MAGIC
# MAGIC Count the number of unique values in each column.

# COMMAND ----------

df.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Display Unique Values
# MAGIC
# MAGIC Display the unique values in the `Sex` column.

# COMMAND ----------

df["Sex"].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Count Category Values
# MAGIC
# MAGIC Count the number of passengers in each `Sex` category.

# COMMAND ----------

df["Sex"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Count Embarked Categories
# MAGIC
# MAGIC Count the number of passengers in each `Embarked` category.
# MAGIC

# COMMAND ----------

df["Embarked"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Count Passenger Classes
# MAGIC
# MAGIC Count the number of passengers in each `Pclass` category.

# COMMAND ----------

df["Pclass"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Check the Minimum Age
# MAGIC
# MAGIC Find the minimum value in the `Age` column.

# COMMAND ----------

df["Age"].min()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Check the Maximum Age
# MAGIC
# MAGIC Find the maximum value in the `Age` column.

# COMMAND ----------

df["Age"].max()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Check the Age Range
# MAGIC
# MAGIC Calculate the difference between the maximum and minimum age.

# COMMAND ----------

df["Age"].max() - df["Age"].min()

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## 16. Inspect a Random Sample
# MAGIC
# MAGIC Display a random sample of five rows from the dataset.

# COMMAND ----------

df.sample(5)

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## 17. Inspect the Last Rows
# MAGIC
# MAGIC Display the last five rows of the dataset.

# COMMAND ----------

df.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Explore the structure, quality, and characteristics of the Titanic dataset.
# MAGIC
# MAGIC - Inspect rows, columns, and data types.
# MAGIC - Check dataset information, missing values, and duplicate rows.
# MAGIC - Identify unique values and category frequencies.
# MAGIC - Examine numerical ranges and sample records.
# MAGIC - Identify basic data-quality issues before further analysis.