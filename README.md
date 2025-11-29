# Hybrid Ensemble Classifier: Forest with SVM

## Project Goal
[cite_start]This project implements a **hybrid classification ensemble** [cite: 155] [cite_start]that uses **Bagging** [cite: 158-160]. [cite_start]The base estimators are randomly selected from two types[cite: 155]:
* [cite_start]**Custom ID3 Tree:** Implemented from scratch[cite: 152].
* [cite_start]**SVM Classifier:** Utilizes an existing library implementation (scikit-learn)[cite: 152].

[cite_start]Final classification is determined by **majority voting** among the estimators[cite: 160, 251].

---

## Architecture and Components

### Custom ID3 Decision Tree
[cite_start]The ID3 algorithm is implemented using the **Information Gain (IG)** criterion [cite: 162-163][cite_start], which is calculated based on **Entropy**[cite: 164].
* [cite_start]The implementation natively handles **discrete attributes**[cite: 275].
* [cite_start]Continuous data (from Breast Cancer and Wine sets) is processed via **discretization (binning)** before being used by ID3 [cite: 276-277].

### Support Vector Machine (SVM)
[cite_start]A library implementation (scikit-learn) with a **linear kernel** is used[cite: 233]. [cite_start]The **One-vs-Rest** strategy is applied for multi-class tasks[cite: 236].

---

## Experimental Plan

[cite_start]The algorithm is tested on three distinct datasets from the UCI repository, chosen to include both discrete and continuous features [cite: 254-255].

### Datasets:
1.  [cite_start]**Mushroom Data Set** (Discrete, 8124 examples) [cite: 256-257].
2.  [cite_start]**Wisconsin Breast Cancer** (Continuous, 569 examples)[cite: 262].
3.  [cite_start]**Wine Quality - Red** (Continuous, 1599 examples, binarized) [cite: 268-269].

### Methodology:
* [cite_start]**Validation:** 5-fold cross-validation is used[cite: 280].
* [cite_start]**Statistics:** Experiments are repeated **25 times** [cite: 279][cite_start]; results are aggregated (mean, standard deviation, best/worst)[cite: 280].
* [cite_start]**Metrics:** Accuracy and Confusion Matrix [cite: 282-283].

### Scenarios Tested:
1.  [cite_start]Influence of **SVM participation** ($p_{svm} \in \{0, 20, 50, 80, 100\}\%$)[cite: 292].
2.  [cite_start]Influence of the **Number of Classifiers** ($T \in \{10, 20, 50, 100\}$)[cite: 294].
3.  [cite_start]Influence of the **SVM regularization parameter $C$**[cite: 295].

---

### Reference Comparison
[cite_start]The performance of the hybrid model is compared against a standard **Random Forest Classifier** to evaluate the impact of the modification[cite: 287]. [cite_start]The custom ID3 implementation is verified against the `DecisionTreeClassifier` in scikit-learn[cite: 285].
