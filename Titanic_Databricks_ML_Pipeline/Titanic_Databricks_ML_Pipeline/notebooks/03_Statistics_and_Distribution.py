# Databricks notebook source
# MAGIC %md
# MAGIC # Statistics and Distribution
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Calculate statistical measures and understand the distribution of numerical data.

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## 1. Load the Dataset
# MAGIC
# MAGIC Load the Titanic dataset into a Pandas DataFrame.

# COMMAND ----------

df = pd.read_csv("/Volumes/workspace/default/ml_data/Titanic-Dataset.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Calculate the Mean
# MAGIC
# MAGIC Calculate the average age of the passengers.

# COMMAND ----------

df["Age"].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Calculate the Median
# MAGIC
# MAGIC Calculate the median age of the passengers.

# COMMAND ----------

df["Age"].median()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Calculate the Mode
# MAGIC
# MAGIC Calculate the most frequently occurring age.

# COMMAND ----------

df["Age"].mode()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Calculate the Variance
# MAGIC
# MAGIC Calculate the variance of passenger ages.

# COMMAND ----------

df["Age"].var()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Calculate the Standard Deviation
# MAGIC
# MAGIC Calculate the standard deviation of passenger ages.

# COMMAND ----------

df["Age"].std()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Generate a Statistical Summary
# MAGIC
# MAGIC Generate a statistical summary of the numerical columns.
# MAGIC

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Calculate Skewness
# MAGIC
# MAGIC Calculate the skewness of passenger ages.

# COMMAND ----------

df["Age"].skew()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Calculate Kurtosis
# MAGIC
# MAGIC Calculate the kurtosis of passenger ages.

# COMMAND ----------

df["Age"].kurt()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Examine the Age Distribution
# MAGIC
# MAGIC Examine how passenger ages are distributed across different age ranges.

# COMMAND ----------

df["Age"].hist(bins=20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Examine the Fare Distribution
# MAGIC
# MAGIC Examine how passenger fares are distributed across different fare ranges.

# COMMAND ----------

df["Fare"].hist(bins=20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Identify the Distribution Shape
# MAGIC
# MAGIC Identify whether the Age and Fare distributions are approximately symmetric or skewed.

# COMMAND ----------

print("Age skewness:", df["Age"].skew())
print("Fare skewness:", df["Fare"].skew())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Summarize the statistical characteristics and distribution of the numerical data.
# MAGIC
# MAGIC - Calculate mean, median, mode, variance, and standard deviation.
# MAGIC - Use `describe()` to obtain a statistical summary.
# MAGIC - Use skewness and kurtosis to understand distribution shape and tail behavior.
# MAGIC - Examine numerical distributions using histograms.
# MAGIC - Identify patterns, spread, and potential extreme values before further analysis.