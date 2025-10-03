# Ensemble Learning Lab

This repository contains a Jupyter notebook that implements and analyzes various ensemble learning techniques using scikit-learn. The lab focuses on combining multiple machine learning models to improve prediction performance, using the Iris and Wine datasets for demonstration.

## Overview

The notebook is divided into four main parts:

- **Part 1: Implementing a Majority Vote Classifier**  
  Builds a custom majority voting ensemble using Logistic Regression, Decision Tree, and K-Nearest Neighbors classifiers on the Iris dataset. Includes data preparation, training, evaluation via cross-validation, and ROC AUC comparisons.

- **Part 2: Bagging - Building Ensemble from Bootstrap Samples**  
  Implements Bagging using Decision Trees on the Wine dataset. Compares performance with individual trees, visualizes decision boundaries, and analyzes the impact of bootstrap sampling.

- **Part 3: AdaBoost - Adaptive Boosting**  
  Applies AdaBoost with Decision Tree stumps on the Wine dataset. Examines the effects of learning rate and number of boosting iterations on error convergence, with visualizations of training and test errors.

- **Part 4: Comparing All Ensemble Methods**  
  Comprehensive comparison of individual classifiers (Logistic Regression, KNN, Decision Tree) and ensembles (Voting Classifier, Bagging, Random Forest, AdaBoost) on the Iris dataset. Includes hyperparameter tuning via GridSearchCV and a detailed analysis report discussing performance, trade-offs, and insights.

The notebook concludes with an **Analysis Report** summarizing key findings, including why ensembles perform better (or not) in certain cases, the impact of hyperparameters, and practical recommendations.

## Requirements

To run the notebook, you'll need:

- Python 3.6+
- Jupyter Notebook or JupyterLab
- Required libraries:
  - scikit-learn
  - numpy
  - pandas
  - matplotlib

Install the dependencies using:

```bash
pip install -r requirements.txt
```

(If you add a `requirements.txt` file, list the packages there. Example content:
```
scikit-learn
numpy
pandas
matplotlib
```)

## Usage

1. Clone the repository:
   ```bash:disable-run
   git clone https://github.com/ttumiso182/Ensemble-Learning-Lab.git
   cd Ensemble-Learning-Lab
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open `ensemble_learning_lab.ipynb` and run the cells sequentially. The notebook includes all necessary code for data loading (from scikit-learn datasets), model training, evaluation, and visualizations.

Note: Some cells may produce warnings (e.g., convergence issues in Logistic Regression or deprecation warnings in AdaBoost). These are noted in the notebook and do not affect the results.

## Datasets

- **Iris Dataset**: Loaded via `sklearn.datasets.load_iris()` (used for majority voting and comprehensive comparisons).
- **Wine Dataset**: Loaded from UCI ML Repository via pandas (used for Bagging and AdaBoost).

## Visualizations

The notebook generates several plots:
- Decision boundaries for Bagging vs. individual Decision Trees.
- Error convergence curves for AdaBoost.
- (Optional) ROC curves and other metrics visualizations.

## Analysis Highlights

- Ensembles generally outperform individual classifiers by reducing bias and variance.
- On simple datasets like Iris, simpler models (e.g., KNN) can match or exceed ensemble performance.
- Key insights on hyperparameters, overfitting, and when to choose each method.

For full details, refer to the "Analysis Report" section in the notebook.

## Contributing

Feel free to fork the repository and submit pull requests for improvements, such as additional ensemble methods (e.g., Gradient Boosting) or more datasets.


## Contact

For questions or suggestions, reach out via GitHub issues or contact the repository owner at [ttumiso182](https://github.com/ttumiso182).
```
