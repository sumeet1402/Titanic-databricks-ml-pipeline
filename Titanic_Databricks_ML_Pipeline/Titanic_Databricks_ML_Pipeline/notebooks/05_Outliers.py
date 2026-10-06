# Databricks notebook source
# MAGIC %md
# MAGIC # Outlier Analysis
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Identify, analyze, and handle potential outliers in the Titanic dataset.

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the Dataset
# MAGIC
# MAGIC Load the Titanic dataset into a Pandas DataFrame.

# COMMAND ----------

df = pd.read_csv("/Volumes/workspace/default/ml_data/Titanic-Dataset.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Understand Outliers
# MAGIC
# MAGIC Identify values that are unusually far from most other values in a dataset.
# MAGIC
# MAGIC An outlier is not automatically an error and should be investigated before removal or modification.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Calculate Q1 and Q3
# MAGIC
# MAGIC Calculate the first quartile (Q1) and third quartile (Q3) of passenger ages.

# COMMAND ----------

Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)

print("Q1:", Q1)
print("Q3:", Q3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Interquartile Range (IQR)
# MAGIC ### FORMULA
# MAGIC IQR = Q3 - Q1
# MAGIC

# COMMAND ----------

IQR = Q3 - Q1

print("IQR:", IQR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Calculate Outlier Boundaries
# MAGIC
# MAGIC Calculate the lower and upper boundaries using the IQR method.
# MAGIC
# MAGIC ###Formula
# MAGIC Lower Bound = Q1 - 1.5 × IQR, 
# MAGIC Upper Bound = Q3 + 1.5 × IQR

# COMMAND ----------

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Lower Bound:", lower)
print("Upper Bound:", upper)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Identify Potential Outliers
# MAGIC
# MAGIC Identify passenger ages that fall outside the IQR boundaries.

# COMMAND ----------

outliers = df[(df["Age"]>upper) | (df["Age"] < lower)]
outliers[["Name", "Age"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Inspect Potential Outliers
# MAGIC
# MAGIC Inspect the passenger details associated with the potential outliers.

# COMMAND ----------

outliers[["Name","Age","Sex","Pclass","Fare"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Evaluate Potential Outliers
# MAGIC
# MAGIC Determine whether the identified outliers represent valid observations or possible data errors.
# MAGIC
# MAGIC Valid unusual values should be retained unless there is evidence that they are incorrect.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Observation
# MAGIC
# MAGIC The identified older ages are plausible passenger ages rather than obvious data errors. Retain these observations for further analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Remove Outliers
# MAGIC
# MAGIC Remove observations that fall outside the IQR boundaries when there is a valid reason to exclude them.

# COMMAND ----------

clean_data = df[(df["Age"] >= lower) & (df["Age"] <= upper)]

clean_data[["Name", "Age"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Replace Outliers
# MAGIC
# MAGIC Replace outlier values with the median when there is a valid reason to reduce the effect of extreme values.

# COMMAND ----------

median_age = df["Age"].median()

replaced_data = df.copy()
replaced_data.loc[replaced_data["Age"] > upper, "Age"] = median_age

replaced_data[["Name", "Age"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Cap Outliers
# MAGIC
# MAGIC Cap values outside the IQR boundaries at the corresponding lower or upper boundary.

# COMMAND ----------

capped_data = df["Age"].clip(lower=lower, upper=upper)

capped_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Transform the Data
# MAGIC
# MAGIC Apply a logarithmic transformation to reduce the influence of large values.
# MAGIC np.log1p(x) calculates log(1 + x) and safely handles zero values.

# COMMAND ----------

log_data = np.log1p(df["Age"])

log_data