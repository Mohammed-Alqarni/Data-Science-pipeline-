# Pipeline Project - README

## Project Overview
This project involves creating a machine learning pipeline to predict whether a product is recommended by a customer based on their review. The dataset provided includes numerical, categorical, and text data, and the task is to preprocess the data, train a model, fine-tune it, and evaluate its performance.

---

## Dataset Description
The dataset contains 8 features and 1 target variable. Below are the details:

### Features:
1. **Clothing ID**: Integer categorical variable referring to the specific product being reviewed.
2. **Age**: Positive integer representing the age of the reviewer.
3. **Title**: String variable representing the title of the review.
4. **Review Text**: String variable containing the review body.
5. **Positive Feedback Count**: Integer representing the number of customers who found the review helpful.
6. **Division Name**: Categorical variable indicating the product's high-level division.
7. **Department Name**: Categorical variable indicating the product's department.
8. **Class Name**: Categorical variable indicating the product's class.

### Target:
- **Recommended IND**: Binary variable where 1 indicates the product is recommended, and 0 indicates it is not recommended.

---

## Project Steps

### 1. Data Preprocessing
- **Numerical Features**: Standardized using `StandardScaler`.
- **Categorical Features**: Encoded using `OneHotEncoder` with `handle_unknown='ignore'`.
- **Text Features**: Processed using `TfidfVectorizer` for both the `Title` and `Review Text` columns.

### 2. Machine Learning Pipeline
- Built using `sklearn.pipeline.Pipeline`.
- Includes a preprocessing step (`ColumnTransformer`) and a classification model (`RandomForestClassifier`).

### 3. Model Training
- Split the data into training and testing sets using an 80-20 split.
- The initial pipeline was trained using default hyperparameters for the Random Forest Classifier.

### 4. Model Evaluation
- Metrics used:
  - **Classification Report**: Precision, recall, F1-score, and support for each class.
  - **ROC AUC Score**: Evaluates the model's ability to distinguish between the two classes.

### 5. Hyperparameter Tuning
- Used `GridSearchCV` to fine-tune the Random Forest Classifier's hyperparameters:
  - Number of estimators (`n_estimators`): [100, 200, 300].
  - Maximum depth (`max_depth`): [10, 20, 30].
  - Minimum samples split (`min_samples_split`): [2, 5, 10].
- Evaluated the model using cross-validation and selected the best-performing parameters.

---

## Results

### Initial Model:
- **Accuracy**: 85%
- **ROC AUC Score**: 0.926

### Fine-Tuned Model:
- **Accuracy**: 83%
- **ROC AUC Score**: 0.933

---

## Requirements
To run this project, ensure the following libraries are installed:

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `category_encoders`
- `spacy`

Install the dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/udacity/dsnd-pipelines-project.git
cd dsnd-pipelines-project
```

### 2. Install Requirements
```bash
python -m pip install -r requirements.txt
```

### 3. Run the Notebook
Open the Jupyter Notebook in your preferred environment and run the cells in sequence.

---

## File Structure
- **reviews.csv**: Dataset file.
- **pipeline_project.ipynb**: Jupyter Notebook containing the pipeline code and analysis.
- **README.md**: This file.

---

## Conclusion
This project successfully demonstrates building, training, and fine-tuning a machine learning pipeline to handle numerical, categorical, and text data effectively. The fine-tuned Random Forest Classifier achieved an impressive ROC AUC score, showing its capability to predict customer recommendations accurately.

---

## Future Improvements
- Experiment with other models (e.g., Gradient Boosting, XGBoost).
- Incorporate advanced text processing techniques (e.g., embeddings like Word2Vec or BERT).
- Address class imbalance with techniques like SMOTE or class weighting.
- Improve hyperparameter tuning using Bayesian Optimization.

---

## Author
This project was completed as part of the Udacity Data Science Nanodegree program.

