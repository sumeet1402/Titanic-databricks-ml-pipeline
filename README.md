# Titanic Databricks ML Pipeline

An end-to-end Machine Learning and Data Engineering project using the Titanic dataset, built with Python, SQL, Databricks, Apache Spark, Delta Lake, Pandas, Scikit-learn, Matplotlib, Seaborn, and MLflow.

The project covers the complete workflow from data collection and exploration to feature engineering, model training, evaluation, experiment tracking, model comparison, and prediction.

Three classification models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest

**Random Forest is currently the best-performing model by test accuracy**, based on the recorded results from the latest model runs.

## Project Overview

The objective is to predict whether a passenger survived the Titanic disaster using passenger information such as passenger class, gender, age, family information, fare, embarkation port, cabin information, and passenger title.

Target variable:

```text
Survived
0 = Did not survive
1 = Survived
```

Since the target contains two classes, this is a binary classification problem.

## Technologies Used

### Programming Languages
- Python
- SQL

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- MLflow

### Data Engineering and Big Data
- Apache Spark
- PySpark
- Databricks
- Delta Lake

### Tools
- Git
- GitHub
- MLflow Tracking

## Project Structure

```text
titanic-databricks-ml-pipeline/
│
├── 01_Data_Collection
├── 02_Data_Exploration
├── 03_Statistics_and_Distribution
├── 04_Visualization
├── 05_Outliers
├── 06_Databricks_Delta_DataLake
├── 07_Feature_Engineering
├── 08_Feature_Encoding
├── 09_Train_Test_Split
├── 10_Model_Training
├── 11_Model_Evaluation
├── 12_Prediction
├── 13_Decision_Tree
├── 14_Random_Forest
├── 15_MLflow_Model_Tracking_and_Comparison
│
├── README.md
└── requirements.txt
```

## Machine Learning Pipeline

```text
Kaggle Titanic Dataset
        |
        v
Data Collection and Exploration
        |
        v
Statistics and Visualization
        |
        v
Outlier Detection
        |
        v
Databricks and Delta Lake
        |
        v
Feature Engineering
        |
        v
Feature Encoding
        |
        v
Train-Test Split
        |
        v
Model Training
        |
        v
Model Evaluation
        |
        v
MLflow Experiment Tracking
        |
        v
Model Comparison
        |
        v
Best Model Selection
        |
        v
Prediction
```

## 1. Data Collection

The Titanic dataset was obtained from Kaggle.

The original dataset contains 891 rows and 12 columns.

Main columns include:

```text
PassengerId
Survived
Pclass
Name
Sex
Age
SibSp
Parch
Ticket
Fare
Cabin
Embarked
```

The dataset was loaded into Databricks for further processing.

## 2. Data Exploration

Initial exploration was performed to understand:

- Dataset shape
- Column names and data types
- Missing values
- Unique values
- Statistical information
- Distribution of categorical variables

Example:

```python
df.shape
df.columns
df.dtypes
df.isnull().sum()
df.describe()
```

## 3. Statistics and Distribution

Statistical analysis included:

- Mean, median, and mode
- Variance and standard deviation
- Minimum and maximum
- Quartiles
- Skewness and kurtosis

Example:

```python
df.describe()
```

This helped explain the distributions of numerical variables such as `Age`, `Fare`, `SibSp`, and `Parch`.

## 4. Data Visualization

Visualizations included:

- Bar charts
- Pie charts
- Histograms
- Box plots
- Scatter plots
- Distribution plots

Examples of analysis:

- Survival by gender
- Passenger class distribution
- Age distribution
- Fare distribution
- Survival across passenger classes

## 5. Outlier Detection

Box plots and the Interquartile Range (IQR) method were used to identify potential outliers.

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Potential outliers were analyzed before proceeding with feature engineering.

## 6. Databricks and Delta Lake

Databricks was used as the development environment for data processing and machine learning.

Delta Lake concepts explored included:

- Delta Tables
- Table History
- Table Versions
- Time Travel
- Data Lake concepts
- Bronze / Silver / Gold architecture

Example:

```sql
SELECT *
FROM default.titanic_clean;
```

## 7. Feature Engineering

New features were created from the existing Titanic data.

### FamilySize

```text
FamilySize = SibSp + Parch + 1
```

### IsAlone

```text
IsAlone = 1 when FamilySize = 1
IsAlone = 0 otherwise
```

### Title

Titles were extracted from passenger names and grouped into:

```text
Mr
Mrs
Miss
Master
Other
```

### AgeGroup

Passengers were grouped into:

```text
Child
Teen
Adult
Senior
```

### FamilySizeGroup

Family sizes were grouped into:

```text
Alone
Small
Medium
Large
```

### Cabin Deck

The first character of the cabin value was extracted to represent the cabin deck. Missing cabin values were grouped as `Unknown`.

## 8. Feature Encoding

Categorical variables were converted into numerical representations.

### Sex

```text
Male = 0
Female = 1
```

### Embarked

One-hot encoded columns:

```text
Embarked_C
Embarked_Q
Embarked_S
```

### Title

```text
Title_Master
Title_Miss
Title_Mr
Title_Mrs
Title_Other
```

Age groups, family-size groups, and cabin decks were also encoded into numerical features.

## 9. Train-Test Split

The dataset was divided into training and testing datasets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

Recorded split sizes:

```text
Training samples: 712
Testing samples: 179
```

The target variable was `Survived`.

## 10. Model Training

### Logistic Regression

Logistic Regression was used as a baseline binary classification model.

```python
LogisticRegression(max_iter=1000)
```

It estimates the probability of a passenger belonging to class `1`. With the usual 0.5 threshold:

```text
Probability >= 0.5 → Prediction = 1
Probability < 0.5  → Prediction = 0
```

### Decision Tree

A Decision Tree learns decision rules from feature values to predict the target class.

### Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

Recorded model configuration:

```python
RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    max_depth=5,
    random_state=42
)
```

## 11. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Accuracy

```text
Correct Predictions / Total Predictions
```

### Precision

```text
True Positives / (True Positives + False Positives)
```

### Recall

```text
True Positives / (True Positives + False Negatives)
```

### F1-score

```text
2 × (Precision × Recall) / (Precision + Recall)
```

## 12. Model Comparison

The following table summarizes the recorded results from the latest model runs.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.12% | 77.63% | 79.73% | 78.67% |
| Decision Tree | 80.40% | 76.71% | 75.68% | 76.19% |
| Random Forest | **82.70%** | **81.20%** | 75.68% | 78.30% |

*Metrics are rounded to two decimal places. These results reflect the recorded runs and should be compared using consistent data, features, and train-test splits.*

## 13. Model Selection

**Random Forest is the current best model by accuracy**, with a recorded test accuracy of approximately 82.70%.

Reasons for considering Random Forest:

- Highest recorded accuracy among the three models.
- Highest recorded precision.
- Combines multiple decision trees to produce predictions.

Logistic Regression achieved higher recall, meaning it identified a larger proportion of actual survivors in the recorded test results. The preferred model ultimately depends on the evaluation metric and project requirements.

Random Forest is the current candidate for further validation and potential deployment.

## 14. MLflow Experiment Tracking

MLflow was introduced to track and compare machine learning experiments in Databricks.

The project logs runs for:

- Logistic Regression
- Decision Tree
- Random Forest

Recorded parameters include model settings and train-test split configuration. Logged metrics include:

- Accuracy
- Precision
- Recall
- F1-score

Model artifacts are also logged for the tracked runs.

MLflow makes it easier to compare experiments, inspect parameters and metrics, and identify promising models without relying only on notebook outputs.

## 15. Prediction

After training, models can generate predictions for passenger data.

For a classifier that supports `predict_proba()`, the output represents the estimated probabilities for the classes.

Example:

```text
[[0.084, 0.916]]
```

This indicates estimated probabilities of:

```text
Probability of not surviving = 8.4%
Probability of surviving     = 91.6%
```

With a 0.5 threshold, the predicted class would be `1`.

Another example:

```text
[[0.936, 0.064]]
```

This produces a predicted class of `0`.

These probabilities are model estimates, not guarantees of an individual outcome.

## Machine Learning Concepts Covered

- Machine Learning lifecycle
- Data collection and exploration
- Data cleaning and missing-value analysis
- Statistical analysis
- Data visualization
- Outlier detection
- Feature engineering and encoding
- Train-test split
- Binary classification
- Logistic Regression
- Decision Trees
- Random Forest
- Model evaluation
- Confusion matrix
- Accuracy, precision, recall, and F1-score
- Prediction probabilities
- Model comparison
- MLflow experiment tracking

## Data Engineering Concepts Covered

- Databricks
- Apache Spark and PySpark
- Spark SQL
- DataFrames
- Transformations and actions
- Lazy evaluation
- Narrow and wide transformations
- Shuffle and DAG
- `explain()`
- Joins and aggregations
- CSV and Parquet
- Delta Lake
- Partitioning
- Schema enforcement and evolution
- Delta Table History
- Time Travel
- Bronze / Silver / Gold architecture

## Architecture

```text
                  Kaggle Dataset
                        |
                        v
                +---------------+
                |    Bronze     |
                |   Raw Data    |
                +-------+-------+
                        |
                        v
                +---------------+
                |    Silver     |
                | Cleaned Data  |
                +-------+-------+
                        |
                        v
                +---------------+
                |     Gold      |
                |  ML Features  |
                +-------+-------+
                        |
                        v
                Feature Encoding
                        |
                        v
                  Train / Test
                     Split
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
     Logistic       Decision      Random
     Regression       Tree        Forest
          |             |             |
          +-------------+-------------+
                        |
                        v
                Model Evaluation
                        |
                        v
                MLflow Tracking
                        |
                        v
                 Model Comparison
                        |
                        v
                  Best Candidate
                        |
                        v
                    Prediction
```

## Final Result

This project demonstrates an end-to-end Machine Learning and Data Engineering workflow using the Titanic dataset.

The project combines data processing in Databricks, Delta Lake storage concepts, feature engineering, classification models, model evaluation, and MLflow experiment tracking.

The three models were trained and compared, with **Random Forest currently leading on recorded test accuracy at approximately 82.70%**.

## Future Improvements

Potential next steps include:

- Hyperparameter tuning
- Cross-validation
- Feature selection
- Model explainability
- Further MLflow experiment comparison
- Model registration and versioning
- Databricks Jobs / Workflows
- Automated ETL pipeline
- Model serving or prediction API
- Streamlit interface and deployment

## Author

**Sumeet Mandhre**

B.Tech Computer Engineering

## Dataset

Titanic dataset from Kaggle.

The dataset is used for educational and Machine Learning practice purposes.
