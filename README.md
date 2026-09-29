# Titanic Survival Prediction Pipeline

An end-to-end machine learning project built using Python and Databricks to predict passenger survival using the Titanic dataset.

## Project Overview

This project follows a complete machine learning workflow, from data collection and exploration to model training, evaluation, and prediction.

The project was developed in Databricks and uses Delta Lake and SQL for data processing and management.

## Workflow

Kaggle Dataset
→ Data Collection
→ Data Exploration
→ Statistics & Distribution
→ Visualization
→ Outlier Handling
→ Delta Lake & Data Lake
→ Feature Engineering
→ Feature Encoding
→ Train/Test Split
→ Model Training
→ Model Evaluation
→ Prediction

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Databricks
- Delta Lake
- SQL

## Machine Learning Model

The project uses **Logistic Regression** to predict whether a passenger survived.

### Model Performance

The model was evaluated on a held-out test set.

- Accuracy: **82.68%**
- Precision: **78.67%**
- Recall: **79.73%**
- F1-Score: **79.19%**

## Prediction

The trained Logistic Regression model can generate both class predictions and survival probabilities using `predict()` and `predict_proba()`.

### Example Predictions

- **Passenger 1:** 91.56% probability of survival
- **Passenger 2:** 6.35% probability of survival

The probability represents the model's estimated probability for each prediction class.

## Feature Engineering

The project creates features such as:

- FamilySize
- IsAlone
- Title
- AgeGroup
- FamilySizeGroup
- Cabin_Deck

Categorical features are converted into numerical features using encoding techniques before model training.

## Databricks & Delta Lake

The project also explores:

- Delta Tables
- Table History
- Table Versions
- Time Travel
- Data Lake concepts
- Bronze → Silver → Gold architecture
- Saving processed data as tables

## Repository Structure

titanic-databricks-ml-pipeline/
│
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   ├── 02_Data_Exploration.ipynb
│   ├── 03_Statistics_and_Distribution.ipynb
│   ├── 04_Visualization.ipynb
│   ├── 05_Outliers.ipynb
│   ├── 06_Databricks_Delta_DataLake.ipynb
│   ├── 07_Feature_Engineering.ipynb
│   ├── 08_Feature_Encoding.ipynb
│   ├── 09_Train_Test_Split.ipynb
│   ├── 10_Model_Training.ipynb
│   ├── 11_Model_Evaluation.ipynb
│   └── 12_Prediction.ipynb
│
├── README.md
└── requirements.txts

## Dataset

The project uses the Titanic passenger dataset containing information such as passenger class, age, sex, family information, fare, cabin, and embarkation point.

## Author

**Sumeet Mandhre**
