# Databricks notebook source
# MAGIC %md
# MAGIC # Data Visualization
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Visualize the Titanic dataset to identify patterns, comparisons, distributions, and relationships between variables.

# COMMAND ----------

import pandas as pd

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
# MAGIC ## 2. Line Plot
# MAGIC
# MAGIC Visualize passenger fares across passenger IDs using a line plot.

# COMMAND ----------

df.plot(x="PassengerId", y="Fare", kind="line")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bar Plot
# MAGIC
# MAGIC Compare the number of passengers in each `Sex` category using a bar plot.

# COMMAND ----------

df["Sex"].value_counts().plot(kind="bar")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Bar Plot for Embarkation Points
# MAGIC
# MAGIC Compare the number of passengers from each `Embarked` category using a bar plot.

# COMMAND ----------

df["Embarked"].value_counts().plot(kind="bar")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Pie Chart
# MAGIC
# MAGIC Visualize the proportion of passengers in each `Sex` category using a pie chart.

# COMMAND ----------

df["Sex"].value_counts().plot(
    kind = "pie",
    autopct = "%1.1f%%"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Box Plot
# MAGIC
# MAGIC Visualize the distribution of passenger ages and identify potential outliers using a box plot.

# COMMAND ----------

df.boxplot(column="Age")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Histogram
# MAGIC
# MAGIC Visualize the distribution of passenger ages using a histogram.

# COMMAND ----------

df["Age"].hist(bins=20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Scatter Plot
# MAGIC
# MAGIC Visualize the relationship between passenger age and fare using a scatter plot.

# COMMAND ----------

df.plot.scatter(x="Age", y = "Fare")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Visualization Summary
# MAGIC
# MAGIC Use appropriate plots to understand different aspects of the dataset.
# MAGIC
# MAGIC - Use line plots to visualize ordered numerical values.
# MAGIC - Use bar plots to compare categorical values.
# MAGIC - Use pie charts to visualize category proportions.
# MAGIC - Use box plots to examine numerical distributions and potential outliers.
# MAGIC - Use histograms to understand the distribution of numerical values.
# MAGIC - Use scatter plots to examine relationships between two numerical variables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Visualize the Titanic dataset using different types of plots to identify patterns, comparisons, distributions, relationships, and potential outliers.