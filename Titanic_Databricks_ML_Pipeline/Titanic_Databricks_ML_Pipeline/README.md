# 🚢 Titanic Survival Prediction — End-to-End Machine Learning Project

## 🛠️ Technologies Used

### Programming Languages

- Python
- SQL

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Platforms & Tools

- Databricks
- Apache Spark
- Delta Lake
- Git
- GitHub

### Machine Learning

- Logistic Regression
- Train/Test Split
- Classification
- Confusion Matrix
- Accuracy

---

## 📂 Project Structure

```text
Titanic-ML-Project/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_Data_Collection
│   ├── 02_Data_Exploration
│   ├── 03_Statistics_and_Distribution
│   ├── 04_Visualization
│   ├── 05_Outliers
│   ├── 06_Feature_Engineering
│   ├── 07_Feature_Encoding
│   ├── 08_Train_Test_Split
│   ├── 09_Model_Training
│   ├── 10_Model_Evaluation
│   └── 11_Prediction
│
└── data/
    └── titanic.csv
```

---

## 🔎 Dataset

The project uses the **Titanic dataset**, containing information about **891 passengers**.

### Important Features

| Feature | Description |
|---|---|
| PassengerId | Unique passenger identifier |
| Survived | Survival status |
| Pclass | Passenger class |
| Name | Passenger name |
| Sex | Passenger gender |
| Age | Passenger age |
| SibSp | Siblings/spouses aboard |
| Parch | Parents/children aboard |
| Fare | Ticket fare |
| Cabin | Cabin information |
| Embarked | Port of embarkation |

---

## 🧹 Data Exploration & Cleaning

The dataset was explored using:

- DataFrame structure and data types
- Missing-value analysis
- Descriptive statistics
- Value counts
- Distribution analysis
- Histograms
- Bar charts
- Boxplots
- Outlier detection

Missing values were handled before building the Machine Learning model.

---

## ⚙️ Feature Engineering

Several new features were created to improve the model.

### Family Size

Family size was calculated using:

```python
FamilySize = SibSp + Parch + 1
```

### Is Alone

A binary feature was created to identify whether a passenger travelled alone.

### Age Group

Passengers were grouped into:

- Child
- Teen
- Adult
- Senior

### Family Size Group

Passengers were categorized into:

- Alone
- Small
- Medium
- Large

### Name Title

Passenger names were used to extract titles such as:

- Mr
- Mrs
- Miss
- Master
- Other

### Cabin Deck

The cabin information was transformed into deck categories.

Missing cabin information was grouped into an `Unknown` category.

---

## 🔢 Feature Encoding

Categorical features were converted into numerical form using **One-Hot Encoding**.

Examples include:

```text
Embarked_C
Embarked_Q
Embarked_S

Title_Master
Title_Miss
Title_Mr
Title_Mrs
Title_Other

AgeGroup_Child
AgeGroup_Teen
AgeGroup_Adult
AgeGroup_Senior
```

This allows categorical information to be used by the Machine Learning model.

---

## ✂️ Train/Test Split

The dataset was divided into training and testing sets.

```text
Training samples: 712
Testing samples: 179
Features: 33
```

The training dataset was used to train the model, while the testing dataset was used to evaluate its performance on unseen data.

---

## 🤖 Machine Learning Model

### Logistic Regression

The final model used in this project is:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
```

Logistic Regression was selected because the target variable is binary:

```text
0 → Did not survive
1 → Survived
```

---

## 📊 Model Evaluation

The model was evaluated using:

- Accuracy
- Confusion Matrix
- Predicted vs Actual values

### Confusion Matrix

```text
[[89, 16],
 [15, 59]]
```

### Accuracy

The model achieved approximately **79–83% accuracy** across the different model runs during development.

The project focuses not only on the final accuracy but also on understanding the complete Machine Learning workflow.

---

## 🗄️ Databricks & Delta Lake

The project also explored important Databricks and Delta Lake concepts, including:

- Catalogs and schemas
- Managed tables
- Delta tables
- Table history
- Table versions
- Time Travel
- Restore operations
- Data Lake concepts
- Medallion Architecture

### Medallion Architecture

The data pipeline follows the structure:

```text
Bronze
Raw Data
   ↓
Silver
Cleaned Data
   ↓
Gold
Feature-Ready Data
```

This approach separates raw, cleaned, and analysis-ready data.

---

## 📈 Key Learning Outcomes

Through this project, I practiced:

- Data collection
- Data exploration
- Data cleaning
- Statistical analysis
- Data visualization
- Missing-value handling
- Outlier detection
- Feature engineering
- Feature encoding
- Train/test splitting
- Logistic Regression
- Model evaluation
- SQL analysis
- Databricks
- Apache Spark
- Delta Lake
- Table versioning
- Time Travel
- Bronze → Silver → Gold architecture

---

## 🚀 Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- Comparing Logistic Regression with Random Forest
- Comparing Logistic Regression with XGBoost
- Precision, Recall and F1-score analysis
- ROC-AUC evaluation
- Feature importance analysis
- Building a prediction interface using Streamlit
- Deploying the model

---

## 👨‍💻 Author

**Sumeet Mandhre**

B.Tech — Computer Engineering

### Interests

- Machine Learning
- Artificial Intelligence
- Data Science
- Python
- Web Development
- UI/UX

---

## ⭐ Project

This project was created as a hands-on learning project to understand and implement an end-to-end Machine Learning workflow using:

**Python · SQL · Databricks · Apache Spark · Delta Lake · Scikit-learn**

The project combines **data engineering, data analysis, feature engineering, and machine learning** into a complete workflow.