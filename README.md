# Hybrid Ensemble Classifier: Forest with SVM

## Project Goal
This project implements a **hybrid classification ensemble** [cite: 155] that uses **Bagging** [cite: 158-160]. The base estimators are randomly selected from two types[cite: 155]:
* **Custom ID3 Tree:** Implemented from scratch[cite: 152].
* **SVM Classifier:** Utilizes an existing library implementation (scikit-learn)[cite: 152].

Final classification is determined by **majority voting** among the estimators[cite: 160, 251].

---

## Architecture and Components

### Custom ID3 Decision Tree
The ID3 algorithm is implemented using the **Information Gain (IG)** criterion [cite: 162-163], which is calculated based on **Entropy**[cite: 164].
* The implementation natively handles **discrete attributes**[cite: 275].
* Continuous data (from Breast Cancer and Wine sets) is processed via **discretization (binning)** before being used by ID3 [cite: 276-277].

### Support Vector Machine (SVM)
A library implementation (scikit-learn) with a **linear kernel** is used[cite: 233]. The **One-vs-Rest** strategy is applied for multi-class tasks[cite: 236].

---

## Experimental Plan

The algorithm is tested on three distinct datasets from the UCI repository, chosen to include both discrete and continuous features [cite: 254-255].

### Datasets:
1.  **Mushroom Data Set** (Discrete, 8124 examples) [cite: 256-257].
2.  **Wisconsin Breast Cancer** (Continuous, 569 examples)[cite: 262].
3.  **Wine Quality - Red** (Continuous, 1599 examples, binarized) [cite: 268-269].

### Methodology:
* **Validation:** 5-fold cross-validation is used[cite: 280].
* **Statistics:** Experiments are repeated **25 times** [cite: 279]; results are aggregated (mean, standard deviation, best/worst)[cite: 280].
* **Metrics:** Accuracy and Confusion Matrix [cite: 282-283].

### Scenarios Tested:
1.  Influence of **SVM participation** ($p_{svm} \in \{0, 20, 50, 80, 100\}\%$)[cite: 292].
2.  Influence of the **Number of Classifiers** ($T \in \{10, 20, 50, 100\}$)[cite: 294].
3.  Influence of the **SVM regularization parameter $C$**[cite: 295].

---

### Reference Comparison
The performance of the hybrid model is compared against a standard **Random Forest Classifier** to evaluate the impact of the modification[cite: 287]. The custom ID3 implementation is verified against the `DecisionTreeClassifier` in scikit-learn[cite: 285].
