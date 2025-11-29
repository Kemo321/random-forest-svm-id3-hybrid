# Hybrid Ensemble Classifier: Forest with SVM

## Project Goal
This project implements a **hybrid classification ensemble** that uses **Bagging**. The base estimators are randomly selected from two types:
* **Custom ID3 Tree:** Implemented from scratch.
* **SVM Classifier:** Utilizes an existing library implementation (scikit-learn).

Final classification is determined by **majority voting** among the estimators.

---

## Architecture and Components

### Custom ID3 Decision Tree
The ID3 algorithm is implemented using the **Information Gain (IG)** criterion, which is calculated based on **Entropy**.
* The implementation natively handles **discrete attributes**.
* Continuous data (from Breast Cancer and Wine sets) is processed via **discretization (binning)** before being used by ID3.

### Support Vector Machine (SVM)
A library implementation (scikit-learn) with a **linear kernel** is used. The **One-vs-Rest** strategy is applied for multi-class tasks.

---

## Experimental Plan

The algorithm is tested on three distinct datasets from the UCI repository, chosen to include both discrete and continuous features.

### Datasets:
1.  **Mushroom Data Set** (Discrete, 8124 examples).
2.  **Wisconsin Breast Cancer** (Continuous, 569 examples).
3.  **Wine Quality - Red** (Continuous, 1599 examples, binarized).

### Methodology:
* **Validation:** 5-fold cross-validation is used.
* **Statistics:** Experiments are repeated **25 times**; results are aggregated (mean, standard deviation, best/worst).
* **Metrics:** Accuracy and Confusion Matrix.

### Scenarios Tested:
1.  Influence of **SVM participation** ($p_{svm} \in \{0, 20, 50, 80, 100\}\%$).
2.  Influence of the **Number of Classifiers** ($T \in \{10, 20, 50, 100\}$).
3.  Influence of the **SVM regularization parameter $C$**.

---

### Reference Comparison
The performance of the hybrid model is compared against a standard **Random Forest Classifier** to evaluate the impact of the modification. The custom ID3 implementation is verified against the `DecisionTreeClassifier` in scikit-learn.
