# Titanic — End-to-End Machine Learning Pipeline

An end-to-end Machine Learning and Data Engineering project using the Titanic dataset, built with Python, PySpark, Databricks, Delta Lake, Pandas, Scikit-learn, Matplotlib, and Seaborn.

The project covers the complete workflow from data collection and exploration to feature engineering, model training, evaluation, model comparison, and prediction.

Three classification models were trained and compared:
- Logistic Regression
- Decision Tree
- Random Forest

## Project Overview

The objective is to predict whether a passenger survived the Titanic disaster based on passenger information such as passenger class, gender, age, family information, fare, embarkation port, cabin information, and passenger title.

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

### Data Engineering and Big Data
- Apache Spark
- PySpark
- Databricks
- Delta Lake

### Tools
- Git
- GitHub

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
│
├── README.md
└── requirements.txt
```

## Machine Learning Pipeline

```text
Kaggle Titanic Dataset
        |
        v
Data Collection
        |
        v
Data Exploration
        |
        v
Statistics and Distribution
        |
        v
Data Visualization
        |
        v
Outlier Detection
        |
        v
Databricks / Delta Lake
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
Model Comparison
        |
        v
Prediction
```

## 1. Data Collection

The Titanic dataset was obtained from Kaggle.

The dataset contains 891 rows and 12 columns.

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
- Column names
- Data types
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

Statistical analysis was performed using:
- Mean
- Median
- Mode
- Variance
- Standard deviation
- Minimum
- Maximum
- Quartiles
- Skewness
- Kurtosis

Example:

```python
df.describe()
```

This helped understand numerical variables such as Age, Fare, SibSp, and Parch.

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

Box plots were used to identify potential outliers.

Outlier analysis was performed using the IQR (Interquartile Range) method.

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Potential outliers were analyzed before proceeding with feature engineering.

## 6. Databricks and Delta Lake

The project was developed using Databricks.

Delta Lake was used for storing and managing processed datasets.

The project explored:
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

```python
FamilySize = SibSp + Parch + 1
```

### IsAlone

```text
IsAlone = 1  when FamilySize = 1
IsAlone = 0  otherwise
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

Family size was grouped into:

```text
Alone
Small
Medium
Large
```

### Cabin Deck

The first character of Cabin was extracted to represent the cabin deck. Missing cabin values were grouped as `Unknown`.

## 8. Feature Encoding

Categorical variables were converted into numerical representations.

### Sex

```text
Male = 0
Female = 1
```

### Embarked

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

### Age Group

```text
AgeGroup_Child
AgeGroup_Teen
AgeGroup_Adult
AgeGroup_Senior
```

Family size groups and cabin decks were also one-hot encoded.

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

Result:

```text
Training samples: 712
Testing samples: 179
```

The target variable was `Survived`.

## 10. Model Training

### Logistic Regression

Logistic Regression was used because Titanic is a binary classification problem.

The model calculates the probability of belonging to class 1.

A threshold of 0.5 is generally used:

```text
Probability >= 0.5 → 1
Probability < 0.5  → 0
```

Example:

```text
Probability = 0.91 → Prediction = 1
Probability = 0.06 → Prediction = 0
```

Model:

```python
LogisticRegression(max_iter=1000)
```

### Decision Tree

A Decision Tree makes predictions using learned decision rules based on feature values.

A simplified example:

```text
Is Sex = Female?
       |
      Yes
       |
Is Pclass <= 2?
       |
      Yes
       |
Prediction = 1
```

### Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

Model configuration:

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
- F1 Score
- Confusion Matrix

### Accuracy

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

```text
Precision =
True Positives /
(True Positives + False Positives)
```

### Recall

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

### F1 Score

```text
F1 =
2 × (Precision × Recall) /
(Precision + Recall)
```

## Model Comparison

Results on the test dataset:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.68% | 78.67% | 79.73% | 79.19% |
| Decision Tree | 80.45% | 76.71% | 75.68% | 76.19% |
| Random Forest | 82.68% | 81.16% | 75.68% | 78.32% |

## Model Selection

For this particular train-test split:
- Logistic Regression and Random Forest achieved the same accuracy: 82.68%.
- Random Forest achieved higher precision.
- Logistic Regression achieved higher recall.
- Logistic Regression achieved a higher F1 score.

Therefore, Logistic Regression was selected as the final model for this project.

Model selection depends on the evaluation metric and business requirement. The choice of Logistic Regression here is based on the results obtained from this particular test split.

## 12. Prediction

After training the final model, predictions were generated for new passenger data.

The model returns probabilities for both classes:

```text
[Probability of 0, Probability of 1]
```

Example:

```text
[[0.084, 0.916]]
```

This means:

```text
Probability of not surviving = 8.4%
Probability of surviving     = 91.6%
```

Since the probability of class 1 is greater than 0.5:

```text
Prediction = 1
```

Another example:

```text
[[0.936, 0.064]]
```

means:

```text
Probability of not surviving = 93.6%
Probability of surviving     = 6.4%
```

Therefore:

```text
Prediction = 0
```

## How Does the Model Predict 0 or 1?

For Logistic Regression:

```text
Passenger Features
       |
       v
Trained Model
       |
       v
Probability of Class 1
       |
       v
Threshold = 0.5
       |
       +----------------------+
       |                      |
    < 0.5                  >= 0.5
       |                      |
       v                      v
       0                      1
```

Therefore:

```text
0 = Passenger predicted not to survive
1 = Passenger predicted to survive
```

## Machine Learning Concepts Covered

- Machine Learning lifecycle
- Data collection
- Data exploration
- Data cleaning
- Missing-value analysis
- Statistical analysis
- Data visualization
- Outlier detection
- Feature engineering
- Feature encoding
- Train-test split
- Classification
- Logistic Regression
- Decision Trees
- Random Forest
- Model evaluation
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1 Score
- Prediction probabilities
- Model comparison

## Data Engineering Concepts Covered

- Databricks
- Apache Spark
- PySpark
- Spark SQL
- DataFrames
- Transformations
- Actions
- Lazy Evaluation
- Narrow and Wide Transformations
- Shuffle
- DAG
- `explain()`
- Joins
- Aggregations
- CSV
- Parquet
- Delta Lake
- Partitioning
- Schema Enforcement
- Schema Evolution
- Delta Table History
- Time Travel
- Data Lake concepts

## Architecture

```text
                  Kaggle Dataset
                        |
                        v
                +---------------+
                |    Bronze     |
                |  Raw Data     |
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
                   Prediction
```

## Final Result

The project demonstrates a complete Machine Learning workflow using the Titanic dataset.

Three classification algorithms were trained and compared:

```text
Logistic Regression
Decision Tree
Random Forest
```

The final selected model for this project was:

```text
Logistic Regression
```

with a test accuracy of:

```text
82.68%
```

Final Result:
| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | **82.68%** | 78.67% | **79.73%** | **79.19%** |
| Decision Tree | 80.45% | 76.71% | 75.68% | 76.19% |
| Random Forest | **82.68%** | **81.16%** | 75.68% | 78.32% |

The project combines Machine Learning, Data Engineering, Databricks, Apache Spark, and Delta Lake into an end-to-end pipeline.

## Future Improvements

Possible future improvements include:
- Hyperparameter tuning
- Cross-validation
- Feature selection
- Handling class imbalance
- Model explainability
- MLflow experiment tracking
- Databricks Jobs / Workflows
- Automated ETL pipeline
- Model deployment
- Prediction API
- Streamlit interface

## Author

**Sumeet Mandhre**

B.Tech Computer Engineering

## Dataset

Titanic dataset from Kaggle.

The dataset is used for educational and Machine Learning practice purposes.
