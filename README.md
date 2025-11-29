# 🌲 Hybrid Ensemble Classifier: Forest with SVM

## 🎯 Project Goal
The project implements a hybrid ensemble classifier, operating on the principles of **Bagging**, where the base estimators are chosen randomly between two types:
* [cite_start]**Custom ID3 Tree:** Implemented from scratch[cite: 158].
* [cite_start]**SVM Classifier:** Utilizes an existing library implementation (scikit-learn)[cite: 158].

[cite_start]Final classification is determined by **majority voting** among the estimators[cite: 166].

---

## 🏗️ Architecture and Components

### 1. Custom ID3 Decision Tree
[cite_start]The ID3 algorithm is implemented greedily, utilizing **Information Gain (IG)** as the splitting criterion[cite: 168, 169]. [cite_start]IG is calculated based on **Entropy**[cite: 171].

### 2. Support Vector Machine (SVM)
[cite_start]The project uses a library implementation of a linear SVM [cite: 239] [cite_start]and applies the **One-vs-Rest** strategy for multi-class problems[cite: 242].

### 3. Ensemble Procedure (Bagging)
[cite_start]The forest is built using the following steps for $T$ estimators[cite: 246]:
1.  [cite_start]A bootstrap sample $D_i$ is drawn with replacement[cite: 164].
2.  [cite_start]An estimator type is chosen based on probability $p_{svm}$ (for SVM) or $1-p_{svm}$ (for ID3)[cite: 162, 163, 250].
3.  [cite_start]The chosen model is trained on $D_i$[cite: 252, 255].

---

## 📊 Experimental Plan

[cite_start]The algorithm is tested on three datasets from the UCI repository, chosen to include both discrete and continuous features[cite: 260, 261].

### Datasets:
1.  [cite_start]**Mushroom Data Set** (Discrete, 8124 examples)[cite: 262, 263].
2.  [cite_start]**Wisconsin Breast Cancer** (Continuous, 569 examples)[cite: 268].
3.  [cite_start]**Wine Quality - Red** (Continuous, 1599 examples, binarized)[cite: 274, 275, 277].

### Methodology:
* [cite_start]**Validation:** 5-fold cross-validation[cite: 287].
* [cite_start]**Statistics:** Each experiment is repeated **25 times** [cite: 285][cite_start], and results are aggregated (mean, standard deviation, best/worst)[cite: 286].
* [cite_start]**Preprocessing:** Continuous attributes are handled via **discretization** (binning) before being processed by the ID3 component[cite: 282, 283].
* [cite_start]**Metrics:** Accuracy and Confusion Matrix[cite: 290, 291].

### Scenarios Tested:
1.  [cite_start]Influence of **SVM participation** ($p_{svm} \in \{0, 20, 50, 80, 100\}\%$)[cite: 293].
2.  [cite_start]Influence of **Number of Classifiers** ($T \in \{10, 20, 50, 100\}$)[cite: 294].
3.  [cite_start]Influence of the **SVM regularization parameter $C$**[cite: 295].
