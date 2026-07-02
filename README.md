# Student Performance Analysis

Exploratory data analysis and machine learning project on a student performance dataset, covering regression, clustering, and classification techniques to understand what factors influence math, reading, and writing scores.

## Overview

This notebook explores a dataset of student exam scores alongside demographic and background attributes (gender, parental level of education, test preparation course, lunch type, race/ethnicity). It combines statistical visualization with supervised and unsupervised machine learning to answer questions such as:

- What predicts a student's math score?
- Do students naturally group into distinct performance clusters?
- Can we classify whether a student passes based on their scores?
- How do scores vary across demographic groups?

## Dataset

The notebook expects a CSV file named `study_performance.csv` with the following columns:

| Column | Description |
|---|---|
| `gender` | Student gender |
| `race_ethnicity` | Race/ethnicity group |
| `parental_level_of_education` | Highest education level of parent(s) |
| `lunch` | Lunch type (standard / free-reduced) |
| `test_preparation_course` | Whether the student completed a test prep course |
| `math_score` | Math exam score |
| `reading_score` | Reading exam score |
| `writing_score` | Writing exam score |

> Place `study_performance.csv` in the same directory as the notebook (or update the file path in the first cell) before running.

## Methods

**1. Linear Regression**
Predicts `math_score` from reading/writing scores, gender, and parental education (one-hot encoded via a `ColumnTransformer` + `Pipeline`). Reports MSE and R², with a predicted-vs-actual scatter plot and a coefficient importance plot.

**2. K-Means Clustering**
Groups students into 3 clusters based on standardized math, reading, and writing scores. Visualized with a 3D scatter plot and a bar chart of cluster centers.

**3. Logistic Regression (Classification)**
Creates a binary `pass`/`fail` label (average score ≥ 60) and trains a classifier on the three scores, evaluated with an ROC curve and AUC.

**4. Exploratory Data Analysis**
- Score distributions (histograms with KDE)
- Boxplots of scores by gender and by race/ethnicity
- Correlation heatmap between the three scores
- Average scores by parental education level
- Average scores by test preparation course
- Pairwise scatter plots of score combinations
- Heatmap of average scores by gender × lunch type
- Violin plot of average score distribution by lunch type

## Tech Stack

- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine learning:** scikit-learn (`LinearRegression`, `LogisticRegression`, `KMeans`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`)

## Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
