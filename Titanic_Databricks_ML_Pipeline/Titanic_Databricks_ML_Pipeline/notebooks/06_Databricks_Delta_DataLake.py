# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks, Delta & Data Lake
# MAGIC
# MAGIC ## 1. Databricks Workspace
# MAGIC
# MAGIC Databricks Workspace is the environment used to organize and work with notebooks, files, folders, and other project resources.
# MAGIC
# MAGIC ### Main components
# MAGIC
# MAGIC - **Workspace** — Organizes notebooks, folders, and workspace resources.
# MAGIC - **Compute** — Provides the processing resources required to run notebooks and workloads.
# MAGIC - **Files** — Used to work with files available in the workspace.
# MAGIC - **Volumes** — Used to store and access data files within the Unity Catalog structure.
# MAGIC - **Serverless** — Databricks-managed compute that can be used without manually managing the underlying infrastructure.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Workspace Structure
# MAGIC
# MAGIC A typical Databricks workflow can be represented as:
# MAGIC
# MAGIC Workspace
# MAGIC ↓
# MAGIC Notebook
# MAGIC ↓
# MAGIC Compute
# MAGIC ↓
# MAGIC Data
# MAGIC ↓
# MAGIC Analysis / Processing

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Catalog, Schema & Tables
# MAGIC
# MAGIC Databricks uses a three-level namespace to organize data:
# MAGIC
# MAGIC catalog.schema.table
# MAGIC
# MAGIC ### Components
# MAGIC
# MAGIC - **Catalog** — Top-level container for organizing schemas.
# MAGIC - **Schema** — Organizes tables, views, and other data objects within a catalog.
# MAGIC - **Table** — Stores structured data.
# MAGIC
# MAGIC ### Example
# MAGIC
# MAGIC workspace.default.titanic_processed
# MAGIC
# MAGIC - `workspace` → Catalog
# MAGIC - `default` → Schema
# MAGIC - `titanic_processed` → Table
# MAGIC
# MAGIC ### Managed vs External Tables
# MAGIC
# MAGIC **Managed Table**
# MAGIC - Databricks manages the table metadata and underlying data storage.
# MAGIC
# MAGIC **External Table**
# MAGIC - Databricks manages the table metadata, while the underlying data remains in an externally specified storage location.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Three-Level Naming
# MAGIC
# MAGIC The general format is:
# MAGIC
# MAGIC catalog.schema.table
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC workspace.default.titanic_processed

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking Catalogs and Schemas
# MAGIC
# MAGIC Databricks SQL can be used to explore the available catalogs and schemas.

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CATALOGS;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW SCHEMAS IN workspace;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN workspace.default;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Three-Level Table Reference
# MAGIC
# MAGIC Tables can be referenced using:
# MAGIC
# MAGIC catalog.schema.table
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC workspace.default.titanic_processed

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM workspace.default.titanic_processed
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SQL Editor
# MAGIC
# MAGIC Databricks SQL Editor is used to query and analyze data using SQL.
# MAGIC
# MAGIC Common SQL operations include:
# MAGIC
# MAGIC - Selecting and filtering data
# MAGIC - Aggregating data
# MAGIC - Grouping and sorting data
# MAGIC - Joining tables
# MAGIC - Using subqueries and CTEs
# MAGIC - Using window functions

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     Pclass,
# MAGIC     COUNT(*) AS passenger_count
# MAGIC FROM workspace.default.titanic_processed
# MAGIC GROUP BY Pclass
# MAGIC HAVING COUNT(*) > 300;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     Sex,
# MAGIC     AVG(Age) AS average_age
# MAGIC FROM workspace.default.titanic_processed
# MAGIC GROUP BY Sex;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Concepts
# MAGIC
# MAGIC `WHERE` filters individual rows before grouping.
# MAGIC
# MAGIC `GROUP BY` creates groups based on one or more columns.
# MAGIC
# MAGIC `HAVING` filters groups after aggregation.
# MAGIC
# MAGIC Window functions perform calculations across related rows while retaining individual rows.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Kaggle → Databricks
# MAGIC
# MAGIC Datasets can be downloaded from Kaggle and uploaded to Databricks for analysis and processing.
# MAGIC
# MAGIC ### Workflow
# MAGIC
# MAGIC Kaggle
# MAGIC ↓
# MAGIC
# MAGIC Download dataset
# MAGIC ↓
# MAGIC
# MAGIC Upload dataset to Databricks
# MAGIC ↓
# MAGIC
# MAGIC Store the file in a Volume
# MAGIC ↓
# MAGIC
# MAGIC Access the file using its Volume path
# MAGIC ↓
# MAGIC
# MAGIC Read the dataset into a DataFrame

# COMMAND ----------

import pandas as pd

df = pd.read_csv("/Volumes/workspace/default/ml_data/Titanic-Dataset.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Volume Path Structure
# MAGIC
# MAGIC The general Volume path follows:
# MAGIC
# MAGIC /Volumes/<catalog>/<schema>/<volume>/<file>
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC /Volumes/workspace/default/ml_data/Titanic-Dataset.csv
# MAGIC
# MAGIC - `workspace` → Catalog
# MAGIC - `default` → Schema
# MAGIC - `ml_data` → Volume
# MAGIC - `Titanic-Dataset.csv` → File

# COMMAND ----------

# MAGIC %md
# MAGIC A Volume is used to store and access files within the Unity Catalog structure. It is different from a Workspace location, which is primarily used for notebooks and workspace resources.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Delta Tables
# MAGIC
# MAGIC Delta Lake is a storage layer used with Databricks that provides reliability and data-management features on top of Parquet data.
# MAGIC
# MAGIC ### Delta Table Structure
# MAGIC
# MAGIC A Delta table consists of:
# MAGIC
# MAGIC - Parquet data files — store the actual data.
# MAGIC - `_delta_log` — stores transaction and table-state information.
# MAGIC
# MAGIC Conceptually:
# MAGIC
# MAGIC Delta Table
# MAGIC
# MAGIC ├── Parquet data files
# MAGIC
# MAGIC └── _delta_log

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parquet vs Delta
# MAGIC
# MAGIC **Parquet**
# MAGIC - File-based columnar storage format.
# MAGIC - Stores data efficiently.
# MAGIC - Does not provide Delta's transaction-management features.
# MAGIC
# MAGIC **Delta**
# MAGIC - Uses Parquet data files.
# MAGIC - Maintains a `_delta_log`.
# MAGIC - Supports ACID transactions, table history, and time travel.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ACID Transactions
# MAGIC
# MAGIC Delta provides ACID transaction guarantees:
# MAGIC
# MAGIC - **Atomicity** — A transaction is completed fully or rolled back.
# MAGIC - **Consistency** — Data remains in a valid state according to defined rules.
# MAGIC - **Isolation** — Concurrent transactions do not improperly interfere with each other.
# MAGIC - **Durability** — Successfully committed changes persist.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL workspace.default.titanic_processed;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Preprocessed Data
# MAGIC
# MAGIC Processed data can be saved as a table so that it can be accessed and reused later.
# MAGIC
# MAGIC The `saveAsTable()` method is used to save a Spark DataFrame as a table.
# MAGIC
# MAGIC ### Basic Syntax
# MAGIC
# MAGIC df.write.saveAsTable("catalog.schema.table")

# COMMAND ----------

spark_df_processed = spark.createDataFrame(df)
spark_df_processed.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "workspace.default.titanic_clean"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Modes
# MAGIC
# MAGIC Different write modes control what happens when the target table already exists:
# MAGIC
# MAGIC - `error` / `errorifexists` — Raises an error if the table exists.
# MAGIC - `overwrite` — Replaces the existing data.
# MAGIC - `append` — Adds new data to the existing table.
# MAGIC - `ignore` — Does nothing if the table already exists.

# COMMAND ----------

# Overwrite existing data
spark_df_processed.write.mode("overwrite").saveAsTable(
    "workspace.default.titanic_clean"
)

# COMMAND ----------

# Append data
spark_df_processed.write.mode("append").saveAsTable(
    "workspace.default.titanic_clean"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Important
# MAGIC
# MAGIC `saveAsTable()` persists the DataFrame as a table in the specified catalog and schema.
# MAGIC
# MAGIC The write mode determines how existing table data is handled.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Table Versions
# MAGIC
# MAGIC Delta tables maintain different versions as transactions are committed.
# MAGIC
# MAGIC Each successful transaction can create a new table version.
# MAGIC
# MAGIC ### Example
# MAGIC
# MAGIC Version 0 → Initial table
# MAGIC Version 1 → Data appended
# MAGIC Version 2 → Data updated
# MAGIC Version 3 → Data deleted
# MAGIC
# MAGIC A version number represents a table state after a committed transaction. It does not represent the number of rows in the table.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Important
# MAGIC
# MAGIC Different operations can create new table versions, including:
# MAGIC
# MAGIC - INSERT / WRITE
# MAGIC - APPEND
# MAGIC - UPDATE
# MAGIC - DELETE
# MAGIC - MERGE
# MAGIC - OVERWRITE
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC Version 4
# MAGIC ↓
# MAGIC UPDATE transaction
# MAGIC ↓
# MAGIC Version 5

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.default.titanic_processed;

# COMMAND ----------

# MAGIC %md
# MAGIC The table history can be used to identify the version number, timestamp, operation, and other details associated with each transaction.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Table History
# MAGIC
# MAGIC Delta provides table history to track changes made to a table.
# MAGIC
# MAGIC ### Command
# MAGIC
# MAGIC DESCRIBE HISTORY workspace.default.titanic_processed;
# MAGIC
# MAGIC The history records information about transactions and table versions.
# MAGIC
# MAGIC ### Important Columns
# MAGIC
# MAGIC - `version` — Identifies the table version.
# MAGIC - `timestamp` — Shows when the transaction occurred.
# MAGIC - `operation` — Shows the operation performed, such as WRITE, UPDATE, or DELETE.
# MAGIC - `operationParameters` — Provides details about the operation.
# MAGIC - `operationMetrics` — Provides measurable results of the operation.
# MAGIC
# MAGIC ### Example
# MAGIC
# MAGIC Version 5
# MAGIC Operation: UPDATE
# MAGIC Operation Metrics: Number of records affected
# MAGIC
# MAGIC This means Version 5 represents the table state after the UPDATE transaction was committed.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.default.titanic_processed;

# COMMAND ----------

# MAGIC %md
# MAGIC Table history helps understand what changes occurred, when they occurred, and which table version was created by each transaction.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Time Travel
# MAGIC
# MAGIC Time Travel allows us to read a previous version of a Delta table without changing the current table.
# MAGIC
# MAGIC There are two common ways to access historical data:
# MAGIC
# MAGIC 1. Version-based Time Travel
# MAGIC 2. Timestamp-based Time Travel

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM workspace.default.titanic_processed
# MAGIC VERSION AS OF 2;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Version-based Time Travel
# MAGIC
# MAGIC `VERSION AS OF` is used when the required table version is known.
# MAGIC
# MAGIC The query reads the table as it existed at that version. It does not change the current table.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM workspace.default.titanic_processed
# MAGIC TIMESTAMP AS OF '2026-09-22 05:02:00';

# COMMAND ----------

# MAGIC %md
# MAGIC ### Timestamp-based Time Travel
# MAGIC
# MAGIC `TIMESTAMP AS OF` is used when the required historical timestamp is known.
# MAGIC
# MAGIC The query returns the table state corresponding to that time without modifying the current table.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time Travel vs Restore
# MAGIC
# MAGIC **Time Travel**
# MAGIC - Reads or views a historical table state.
# MAGIC - Does not modify the current table.
# MAGIC
# MAGIC **Restore**
# MAGIC - Makes a previous table state the current state.
# MAGIC - The restore itself creates a new table version.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Data Lake
# MAGIC
# MAGIC A Data Lake is a centralized storage environment that can store large amounts of data from different sources and in different formats.
# MAGIC
# MAGIC Data can come from:
# MAGIC
# MAGIC - CSV and Excel files
# MAGIC - Databases
# MAGIC - APIs
# MAGIC - Application logs
# MAGIC - JSON and Parquet files
# MAGIC - Other data sources

# COMMAND ----------

# MAGIC %md
# MAGIC ### Medallion Architecture
# MAGIC
# MAGIC Data is commonly organized into three logical layers:
# MAGIC
# MAGIC 🥉 Bronze → Raw data
# MAGIC 🥈 Silver → Cleaned and transformed data
# MAGIC 🥇 Gold → Business and analysis-ready data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bronze Layer
# MAGIC
# MAGIC The Bronze layer contains raw or minimally processed data.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC A raw Titanic CSV downloaded from Kaggle can be stored in the Bronze layer.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Silver Layer
# MAGIC
# MAGIC The Silver layer contains cleaned and transformed data.
# MAGIC
# MAGIC Examples:
# MAGIC
# MAGIC - Handling missing values
# MAGIC - Cleaning data
# MAGIC - Removing unnecessary columns
# MAGIC - Creating derived columns
# MAGIC - Transforming data types

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gold Layer
# MAGIC
# MAGIC The Gold layer contains business or analysis-ready data.
# MAGIC
# MAGIC Examples:
# MAGIC
# MAGIC - Aggregated results
# MAGIC - Business metrics
# MAGIC - Analytical datasets
# MAGIC - Dashboard-ready data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delta and Data Lake
# MAGIC
# MAGIC Delta tables can be used within the different layers of a Data Lake.
# MAGIC
# MAGIC Bronze → Delta table
# MAGIC Silver → Delta table
# MAGIC Gold → Delta table
# MAGIC
# MAGIC Delta provides features such as ACID transactions, table history, and time travel, while the Bronze, Silver, and Gold layers organize data according to its processing stage.