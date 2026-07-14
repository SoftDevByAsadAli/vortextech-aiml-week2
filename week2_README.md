# VortexTech AI/ML Internship – Week 2: Build a Classification Model

## What this is
Week 2 submission for the VortexTech AI & ML Internship Track. The task: train a binary
classification model with scikit-learn and evaluate it with proper metrics.

**Dataset used:** Titanic Passenger Dataset (same dataset as Week 1)
**Target variable:** `Survived` (0 = did not survive, 1 = survived)

## What was done
- Re-applied the Week 1 cleaning steps (median-fill `Age`, drop `Cabin`, mode-fill
  `Embarked`) so the notebook runs standalone
- Selected 7 feature columns (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`)
  and one-hot encoded the categorical ones (`Sex`, `Embarked`) with `pd.get_dummies()`
- Split the data 80/20 into training and test sets (`train_test_split`, `random_state=42`)
- Trained two models for comparison:
  - **Logistic Regression** → Accuracy 0.810, Precision 0.786, Recall 0.743, F1 0.764
  - **Decision Tree** (max_depth=5) → Accuracy 0.799, Precision 0.828, Recall 0.649, F1 0.727
- Built a confusion matrix for the Logistic Regression model
- Wrote a markdown summary interpreting the results, the model's limitations, and next steps
  (feature engineering, cross-validation, trying Random Forest/Gradient Boosting)

## Files
- `week2_classification_model.ipynb` — the full notebook with markdown explanations
- `titanic.csv` — the dataset used (same source as Week 1)
- `README.md` — this file

## How to run it
1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/vortextech-aiml-week2.git
   cd vortextech-aiml-week2
   ```
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn jupyter
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook week2_classification_model.ipynb
   ```
4. Run all cells (`Cell → Run All`) to reproduce the training, evaluation, and comparison.

## Author
Asad Ali — VortexTech AI & ML Internship Track, Week 2
