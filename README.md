# Project: Predicting Product Recommendations

This project predicts whether a product will be recommended based on customer reviews using a machine learning pipeline. The data includes features like text reviews, categorical information, and numerical values.

---

## Key Highlights

### Data Preparation
- Loaded the `reviews.csv` dataset and split it into features (`X`) and target (`y`).
- Performed an 80/20 train-test split.

### Pipeline Setup
- Preprocessed numerical data using `StandardScaler`.
- Categorical data encoded with `OneHotEncoder`.
- Textual data transformed into numerical features using `TF-IDF Vectorizer` (limiting to 1000 features, removing stop words).
- Combined these steps into a `ColumnTransformer` and linked it to a `RandomForestClassifier` using a pipeline.

### Why TF-IDF?
TF-IDF was used to convert textual data into a numerical format by capturing the importance of words in the reviews. This helps in identifying key terms influencing recommendations.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Example: Applying TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
text_data_tfidf = vectorizer.fit_transform(text_data)  # text_data contains the review texts
print(text_data_tfidf.shape)  # Output: (number_of_samples, 1000)
```

### Model Training and Evaluation
- Trained the pipeline on the training set.
- Achieved an initial ROC AUC score of **0.926**, indicating good model performance.

### Hyperparameter Tuning
- Fine-tuned the `RandomForestClassifier` using `GridSearchCV` with parameters like `n_estimators` and `max_depth`.
- Improved the ROC AUC score to **0.933**.

### Model Performance
- The initial model had a good weighted F1-score of **0.80**, with precision and recall values showing a strong ability to predict positive cases. However, the recall for class 0 was relatively low, which might indicate difficulty in identifying non-recommendations.
- After fine-tuning, the recall for class 0 decreased further, but the overall ROC AUC score improved slightly, suggesting that the model prioritizes correct predictions of recommendations (class 1).

---

## How to Run
1. Install the required libraries: `pip install pandas scikit-learn`.
2. Place `reviews.csv` in the project directory.
3. Execute the script to train and evaluate the model.
4. Review evaluation metrics and the tuned parameters in the console.

