# Wine Type and Quality Classification

## Content
- Abstract
- Data Preprocessing
- Wine Type Classification
- Wine Quality Classification
- Result and Discussion

# Abstract

本文所使用的資料來自 UCI Machine Learning Repository 中的  [**Wine Quality Data Set**](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)，這個葡萄酒資料集的來源是來自於葡萄牙西北部的綠酒 Vinho Verde。資料集內加上酒的品質共包含了 12 個變量 (Fig. 1)，因為資料集多個變數存在右偏的特性，所以在比較 Log 和 Yeo-Johnson 轉換後選擇了後者 (Table 1 & 2)，Fig. 2 中顯示了在轉換後的資料分布，並用顏色區別紅白酒 (酒紅色為紅酒，米白色為白酒)。

本次做分類學習的步驟為先使用 Cross Validation 選出最適合這個資料集和演算法的參數，我採用的是 5-Folds CV 加上 GridsearchCV 以調整超參數的方法比較每個參數組合的表現並選出每個演算法的最佳參數，再以 Sequential Feature Selector (SFS) 和 Lasso 共同選出資料集中重要的 Features 後並用這組 Feature Group 去做學習，並以 Confusion Matrix 和 ROC curve 評估模型表現，最後再以改變 Model Complexity 的方式去看其是否會有 Overfitting 的問題。

在做葡萄酒種類分類時，我比較了七個模型，最後以 XGBoost 的準確率 99.46% 為最高，Logistic Regression, LDA, QDA 的準確率也達到 99% 以上，且在預測白酒的部分 Accuracy 均較紅酒高，在比較 Feature Importance 後發現 ”total sulfur dioxide” , “chlorides” & “volatile acidity” 三個為各模型中最重要的三個變量。

在做葡萄酒品質分類時，我參考了 P. Cortez *et al.* <sub>[1]</sub> 在 2009 年發表的 “Modeling wine preferences by data mining from physicochemical properties” 中的結果顯示 SVC 在他的結果中具有最高的 Accuracy，不同於 P. Cortez *et al.* 改變 Error tolerence 的做法，我嘗試將資料集 Upsample 以解決其集中在 Quality 5~6 的部分，並成功使 Classification Accuracy 提高到 82.39% 和 78.22%。 

$$
Cross\ Validation \rightarrow Feature\ Selection\rightarrow Confusion\ Matrix\ \\& \ ROC \rightarrow Model\ Complexity\ Test
$$

<img width="1259" alt="截圖 2023-04-30 16 10 20" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/12f6f175-f814-4b61-b269-375f13adb56b">

Fig. 1 [1]

# Data Preprocessing

從 Table 1 & 2 可以觀察到資料在轉換前存在著很大的 Skewness 和 Kurtosis，特別是 “chlorides” 的在紅白酒的偏度都超過 5，紅酒的峰度更是高達 41，在 Log 轉換後，變數普遍表現得不錯，但 “residual sugar” , “chlorides” , “sulphates” 仍不夠接近於常態分佈，但在經過 Yeo-Johnson 轉換後，紅白酒的偏度絕對值都能落在 0.15以下，峰度除了少數變量以外也幾乎都能落在 1 以下，相較 Log 的表現更為出色，故本文將選擇 Yeo-Johnson 做為資料的轉換模式 (Fig. 2-1~11)。

### Red Wine

<img width="1144" alt="截圖 2023-05-03 21 34 09" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/4ffb6516-3b59-466c-9537-2c7ee0edbbb8">

Table 1

### White Wine

<img width="1142" alt="截圖 2023-05-03 21 41 57" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/7656446b-1bba-4c7e-85cb-6799f3d2d971">

Table 2
另外，在轉換後的變數相關性測試中 (Fig. 3) “total sulfur dioxide” 和 “free sulfur dioxide” 兩者有最高的 0.72，似乎不太意外，因為 “total sulfur dioxide” 是指紅白酒中所含有的二氧化硫 $SO_2$  總量，包含未鍵結的 “free sulfur dioxide”，所以兩者具有高度正相關這點是在預料之中的。做 Variance Inflation Factor 後可以發現 “density”的值高達 15 (Table 3-1)，表示有共線性問題 <sub>[2]</sub>，故將其刪除後即可使全部變量的 VIF 都低於 5 (Table 3-2)。

### Correlation test

<img width="500" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/252f21e9-7358-4e64-9e93-ac70e6ac73ef">

Fig. 3-1

### Variance Inflation Factor

<img width="200" alt="截圖 2023-05-01 17 56 18" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/3acecd93-2163-482b-b4a0-e2e8b0ee9de5">
<img width="200" alt="截圖 2023-05-01 17 57 04" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/4f4b981e-e3b3-447c-8ffd-d3d4d22cb005">

Table 3-1 & Table 3-2

---

# Wine Type Classification

先說結論，在 Wine type Classification 用來分類的七個演算法當中，所有的 Accuracy 都達到了 95% 以上，甚至 Logistic Regression, LDA, QDA, XGBoost 都達到 99% 以上，而最高的是 XGBoost 的 99.46%。且在所有演算法中，預測白酒的精準度都比預測紅酒來得高，我認為這非常合理，因為兩種類的資料集大小本身存在差異 (白酒: 4898 ; 紅酒: 1599)，有更多的資料可以針對白酒做學習，自然精準度也會較高。

另外，在所有模型中，”total sulfur dioxide” , “chlorides” & “volatile acidity” 均為 Feature Importance 的前三名，其分別對葡萄酒種類在 95% Confidence interval 下的機率分佈如 Fig. 4-1~3 (Wine = 1 紅酒 ; Wine = 0 白酒)，而 Feature Importance 相對非常低的 “residual sugar” , “alcohol” , “free sulfur dioxide” 分佈如 Fig. 4-4 ~ 6。可以觀察到重要的變量其分佈都是非常鮮明的，處在 0 < P < 1 的區間非常小，且由 Confidence interval 可以看到其標準差都很小。反觀三組較不重要的變數，”residual sugar” 和 “alcohol” 均只有分佈在白酒的區域，也可以看到這兩個變量在紅酒的部分幾乎沒有一個明顯的分佈，且峰度分別低達 -1.08 和 -0.93 (Table 2)。而 “free sulfur dioxide” 則是約有 1/3 的區間在 0 < P < 1 的區域，且從 Confidence interval 可以發現其標準差非常的大。

以下為各演算法的個別表現，內容主要會包含四個部分

1. Grid Search Cross Validation: 以參數調整的方式找出該演算法表現最佳的參數
2. Feature Selection: 利用 Stepwise Selection 選出重要性較高的變量，以減少變量的方式降低 Overfitting 的可能性和雜訊的產生
3. Model Performance: 在選完變數後即開始以該演算法進行學習，表現評估的部分包括 Confusion Matrix, Accuracy, Precision, Recall, F1 score, ROC curve 和 AUC，最後以該模型最重要的兩個變量繪製 Decision Region 以觀察模型的分類方法
4. Model Complexity: 最後利用改變模型複雜度以驗證是否有 Overfitting 的現象，若有 Overfitting 的現象產生時，會出現如 KNN model 在減少 n_neighbors 時出現的 test error 突然高起的狀況，但這個狀況在這次的學習中幾乎沒有發生。

| Table 3 | Logistic Regression | LDA | QDA | KNN | Naive Bayes | SVC | XGBoost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.9905 | 0.9906 | 0.9931 | 0.9517 | 0.9868 | 0.9834 | 0.9946 |

<img width="1283" alt="截圖 2023-09-17 13 59 03" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/9ffb7f1e-55b2-4002-a4f7-7f845df095f1">

## Lasso Regression as Feature Selection

```python
importance > 0.01

['volatile acidity', 'free sulfur dioxide', 'sulphates', 'quality']
```

# Logistic Regression

**Cross Validation**

```python
params = {
    'C': [ 10**i for i in range(-4, 5) ],
    'penalty': ['l1', 'l2', 'elasticnet', 'None'],
    'solver': ['lbfgs', 'liblinear', 'saga']   
}

Best params: {'C': 10000, 'penalty': 'l2', 'solver': 'lbfgs'}
Best scores: 0.9905
```

**Feature Selection**

```python
SFS = ['volatile acidity', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
Lasso = ['volatile acidity', 'free sulfur dioxide', 'sulphates', 'quality']

Feature Group = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'quality']
```
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/e527ebe6-3d67-4237-9afd-73ab3f4a4273">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/514c261b-4322-4499-aa93-d1295ca6a303">

Fig. 6-1 & Fig. 6-2

**Model Performance**

|  | Predict White | Predict Red |
| --- | --- | --- |
| True White | 0.9949 | 0.0050 |
|  True Red | 0.0161 | 0.9839 |

$$
Acurracy:  99.2000 \\%
$$

$$
Precision: 99.1995 \\%
$$

$$
Recall: 99.2000 \\%
$$

$$
f1: 99.1997 \\%
$$

**Model Complexity**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/4bcda33e-f7ac-469d-907c-d40017fc2c3f">

Fig. 6-3

---

# Linear Discriminant Analysis

**Cross Validation**

```python
shrinkage_range = [10**i for i in range(-10, 1)]

params = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': ['auto', None, shrinkage_range],
}

Best params: {'shrinkage': 'auto', 'solver': 'lsqr'}
Best scores: 0.9906
```

**Feature Selection**

```python
SFS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']
Lasso = ['fixed acidity', 'volatile acidity', 'chlorides', 'pH', 'sulphates', 'alcohol']

Feature Group = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']
```

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/a5175e90-7d45-4d06-981d-c5b1c1fb97e4">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/3e56fda4-781a-44f6-bbda-4aedbb3c20ad">

Fig. 7-1 & Fig. 7-2

|  | Predict White | Predict Red |
| --- | --- | --- |
| True White | 0.9924 | 0.0076 |
|  True Red | 0.0230 | 0.9970 |

$$
Acurracy:  98.8308 \\%
$$

$$
Precision: 98.8300 \\%
$$

$$
Recall: 98.8308 \\%
$$

$$
f1: 98.8308 \\%
$$

**Model Complexity**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/94d47adb-acea-4d55-b11e-254828be31ec">

Fig. 7-3

---

# Quadratic Discriminant Analysis

**Cross Validation**

```python
reg_param = [10**i for i in range(0, -10, -1)]

params = {
    'reg_param': reg_param,
}

Best params: {'reg_param': 1e-07}
Best scores: 0.9931
```

**Feature Selection**

```python
SFS = ['volatile acidity', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'sulphates']
Lasso = ['fixed acidity', 'volatile acidity', 'chlorides', 'pH', 'sulphates', 'alcohol']

Feature Group = ['volatile acidity', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'sulphates', 'fixed acidity', 'pH', 'alcohol']
```

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/dc033b92-08b4-4af0-bae5-a702f8bb95f5">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/a8a68668-17e8-42cf-b544-0924bb152e62">

Fig. 8-1 & Fig. 8-2

**Model Performance**

|  | Predict White | Predict Red |
| --- | --- | --- |
| True White | 0.9966 | 0.0034 |
|  True Red | 0.0138 | 0.9862 |

$$
Acurracy:  99.3846 \\%
$$

$$
Precision: 99.3841 \\%
$$

$$
Recall: 99.3846 \\%
$$

$$
f1: 99.3842 \\%
$$

**Model Complexity**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/e823b215-5d99-4844-8096-5f0d0e726fe0">

Fig. 8-3

---

# K Nearest Nieghbors

**Cross Validation**

```python
params = {
    'n_neighbors': list(range(1, 11)),
    'leaf_size': list(range(10, 41, 5)),
    'p': [1, 2],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

Best params: {'leaf_size': 10, 'metric': 'manhattan', 'n_neighbors': 9, 'p': 1, 'weights': 'distance'}
Best scores: 0.9517
```

**Feature Selection**

```python
SFS = ['chlorides', 'total sulfur dioxide', 'alcohol']
Lasso = ['fixed acidity', 'volatile acidity', 'chlorides', 'pH', 'sulphates', 'alcohol']

Feature Group = ['fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']
```

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/03665ae2-c23f-4085-aa9d-be92b6b8ad78">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/804c6201-7905-4bfe-be53-96a9e43420bc">

Fig. 9-1 & Fig. 9-2

**Model Performance**

|  | Predict White | Predict Red |
| --- | --- | --- |
| True White | 0.9966 | 0.0034 |
|  True Red | 0.0115 | 0.9885 |

$$
Acurracy:  99.4462 \\%
$$

$$
Precision: 99.4458 \\%
$$

$$
Recall: 99.4462 \\%
$$

$$
f1: 99.4460 \\%
$$

**Model Complexity**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/b5d7c863-787f-4389-be5d-35e0e079c3aa">

Fig. 9-3

---

# Gaussian Naive Bayes

**Cross Validation**

```python
var_smoothing = [10**i for i in range(0, -15, -1)]

params = {
    'var_smoothing': var_smoothing
}

Best params: {'var_smoothing': 1e-10}
Best scores: 0.9868
```

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/1d8b1edf-8258-47d5-b3c5-73f4d835cedd">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/8aafc3e6-d2f8-4d84-9a74-951be6881aff">

Fig. 10-1 & Fig. 10-2

**Model Performance**

|  | Predict White | Predict Red |
| --- | --- | --- |
| True White | 0.9966 | 0.0034 |
|  True Red | 0.0277 | 0.9723 |


$$
Acurracy:  99.0154 \\%
$$

$$
Precision: 99.0157 \\%
$$

$$
Recall: 99.0154 \\%
$$

$$
f1: 99.0125 \\%
$$

**Model Complexity**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/05480775-6d7b-41ce-842a-ad2c117872f4">

Fig. 10-3

---

# Random Forest

**Cross Validation**

```python
params = {
    'n_estimators': list(range(100, 501, 50)),
    'max_depth': [1, 3, 5, 7, 9],
    'max_features': ['sqrt', 'log2'],  
    'criterion': ['gini', 'entropy'] 
}

Best params: {'criterion': 'entropy', 'max_depth': 9, 'max_features': 'log2', 'n_estimators': 250}
Best scores: 0.9923
```

**Feature Selection**

```python
SFS = ['fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide', 'sulphates']
Lasso = ['fixed acidity', 'volatile acidity', 'chlorides', 'pH', 'sulphates', 'alcohol']

Feature Group = ['fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide', 'sulphates', 'pH', 'alcohol']
```

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/515d97d2-b782-44ed-a678-4c52785865ca">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/afa49b47-f301-44fb-9d6d-c7601ebe8ac0">

Fig. 11-1 & Fig. 11-2


**Model Performance**

|  | Predict White | Predict Red |
| --- | --- | --- |
| True White | 0.9991 | 0.0008 |
|  True Red | 0.0161 | 0.9839 |


$$
Acurracy:  99.5077 \\%
$$

$$
Precision: 99.5090 \\%
$$

$$
Recall: 99.5077 \\%
$$

$$
f1: 99.5066 \\%
$$

**Model Complexity**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/6a9d9ab4-079c-4833-9099-e4ec485318d4">

Fig. 11-3

**Decision Tree**
![Rfc decision tree](https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/02ce8eb8-b757-4d6c-81e5-02e342d81834)

Fig. 11-4 越藍表示被分類到 Red Wine 的機率越大 ; 越橘表示被分到 White Wine 的機率越大

---

# Support Vector Classifier

**Cross Validation**

```python
params = {
    'C': np.logspace(-3, 3, num = 7, base = 10),
    'kernel': ['rbf', 'sigmoid'],
    'gamma': np.logspace(-3, 3, num = 7, base = 10),  
}

Best params: {'C': 1000.0, 'gamma': 0.01, 'kernel': 'rbf'}
Best scores: 0.9834
```

**Feature Selection**

```python
SFS = ['volatile acidity', 'chlorides', 'total sulfur dioxide', 'sulphates', 'free sulfur dioxide']
Lasso = ['fixed acidity', 'volatile acidity', 'chlorides', 'pH', 'sulphates', 'alcohol']

Feature Group = ['fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide', 'sulphates', 'pH', 'alcohol', 'free sulfur dioxide']
```

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/4f1c6c60-39c7-41f6-9ed8-c45cd3e2acdb">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/59c31050-d25e-41eb-b8e0-502575e3ef0f">

Fig. 12-1 & Fig. 12-2

**Model Performance**

|  | Predict White | Predict Red |
| --- | --- | --- |
| True White | 0.9983 | 0.0017 |
|  True Red | 0.0184 | 0.9816 |

$$
Acurracy:  99.3846 \\%
$$

$$
Precision: 99.3854 \\%
$$

$$
Recall: 99.3846 \\%
$$

$$
f1: 99.3832 \\%
$$

**Model Complexity**

<img width="549" alt="截圖 2023-05-01 02 35 40" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/da8241f2-26cd-4372-a13b-1fd77d52d15b">

Fig. 12-3

---

# XGBoost

**Cross Validation**

```python
params = {
    'max_depth': [1, 3, 5, 7, 9, 11],
    'n_estimators': [50, 100, 200, 250, 300],
}

Best params: {'max_depth': 3, 'n_estimators': 250}
Best scores: 0.9890
```

**Feature Selection**

```python
SFS = ['fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide', 'pH', 'sulphates']
Lasso = ['fixed acidity', 'volatile acidity', 'chlorides', 'pH', 'sulphates', 'alcohol']

Feature Group = ['fixed acidity', 'volatile acidity', 'chlorides', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']
```

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/f4bc910f-a754-48a3-93b9-3e90e6d60b7a">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/8b084bf6-bcb6-45ba-997f-45fde8a16ed9">

Fig. 13-1 & Fig. 13-2

**Model Performance**

|  | Predict White | Predict Red |
| --- | --- | --- |
| True White | 0.9983 | 0.0017 |
|  True Red | 0.0161 | 0.9839 |


$$
Acurracy:  99.4462 \\%
$$

$$
Precision: 99.4465 \\%
$$

$$
Recall: 99.4462 \\%
$$

$$
f1: 99.4451 \\%
$$

**Model Complexity**

<img width="504" alt="截圖 2023-05-01 02 32 13" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/7984f676-c0cc-4fd2-9e9d-47905e0f3ef5">

Fig. 13-3

**Decision Tree**

<img width="1924" alt="截圖 2023-05-02 13 30 58" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/02c3db27-9488-45a9-af68-7ddd24f7f231">

Fig. 13-4

{'fixed acidity'} {0} ; {'volatile acidity'} {1} ; {'chlorides'} {2} ; {'total sulfur dioxide'} {3} ; {'pH'} {4} ; {'sulphates'} {5} ; {'alcohol'} {6}

---

# Wine Quality Classification

參考了 P. Cortez *et al.* 的文章結論顯示，SVC 在預測葡萄酒品質的表現是最好的 (相較於 Multiple Regression 和 Neural Networks)，故本文也以 SVC model 做 wine quality 的學習。文章中是以增加 Error tolerance 的方式使 Classification accuracy 從 43.2% 增加到 89.0%，SVC 中的誤差項通過設置誤差上限 $\epsilon$ 來限制，使得預測值和實際值之間的絕對差值小於或等於 $\epsilon$，通過調整 $\epsilon$ 的值，我們可以控制 SVC 模型的準確性。較小的 $\epsilon$ 值意味著模型對誤差的容忍度更嚴格，這容易導致 Overfitting，而較大的 $\epsilon$  值會導致 Underfitting。簡單來說，SVC 使我們能夠靈活地定義我們的模型可以接受多少誤差，並找到合適的直線或更高維度的超平面來擬合數據 <sub>[3], [4]</sub> 。有鑒於 Wine Quality 呈非常漂亮的常態分佈 (Fig. 14)，但也代表著將在預測等級低的 3, 4 或 等級高的 8, 9 時，將因為資料數量不夠模型去做訓練的關係而導致預測失準，所以我的方法是用 SMOTE (Synthesized Minority Oversampling Technique, sampling strategy = “auto”) 來增加少數類別樣本，SMOTE 的採樣模式是通過在現有的少數類別樣本之間進行插值，為少數類別生成合成樣本。具體來說，它隨機選擇一個少數群體的樣本，並在特徵空間 (feature space) 中找到它的 k 個最近的鄰居 (k nearest neighbors)，然後隨機選擇這些鄰居中的一個，並在連接原始少數群體樣本和所選鄰居的線段上隨機選擇一個點，生成一個新的合成樣本 <sub>[5], [6]</sub>

不同於文章中的做法，我利用 Grid Search CV 找出 SVC 對此資料集的最佳 Error tolerence (倒數正比於 SVC 中的 C) 和 gamma 值再比較有無使用 SMOTE 方法來上採樣的結果比較，包括 Confusion Matrix, Accuracy, Precision, Recall, F1 score, ROC curve 和 AUC，最後觀察其 Model Complexity 判斷是否有 Overfitting 的狀況。

![newplot (24)](https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/2653e0ed-745f-4e34-8933-a4968d6464d6)

Fig. 14

# Support Vector Classifier for Wine Quality

可以從 Table 4 這張簡單的圖表中看到，在 SMOTE upsample 之前 Accuracy 分別只有 52.5% / 61.14%，但在 SMOTE upsample 後 Accuracy 分別提升了 1.57 和 1.28倍 來到 82.39% / 78.22%，且從 Fig. 15-3, 4 , Fig. 16-3, 4 的 Confusion Matrix 可以看到，在 SMOTE 前因為資料集中在 Quality 5~6，導致 Red wine 和 White wine 的分類幾乎也集中在 Quality 5~6 的部分，但在 SMOTE 過後，可以看到分類結果呈現了很好的對角線，即 True Positive。

在觀察 Model Complexity 的部分 (Fig. 15-7, 16-7)可以看到 Training error 大致隨著 gamma 和 C 值增加而降低，但紅白酒的 Testing error 都在 gamma 和 C 太大時會有一個上升趨勢，和 Training error 有一個背離的現象，代表有 Overfitting 的狀況。除此之外，在比較 Before / After SMOTE 的參數選擇和 Model Complexity 的部分 (Fig. 15-7, 16-7) 也可以看到 SVC 模型在 C 和 gamma 之間的 Trade-off 行為，我認為十分有趣 ！

| Table 4 | Red Wine |  | White |  |
| --- | --- | --- | --- | --- |
|  | Before SMOTE | After SMOTE | Before SMOTE | After SMOTE |
| Accuracy | 52.50% | 82.39% | 61.14% | 78.22% |
| Precision | 73.24% | 81.84% | 78.71% | 89.88% |
| Recall | 52.50% | 82.39% | 61.14% | 78.22% |
| F1 score | 44.88% | 81.87% | 56.30% | 80.47% |

| TOP 3 Features | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Red wine | “total sulfur dioxide” | “free sulfur dioxide” | “citric acid” |
| White wine | “total sulfur dioxide” | “free sulfur dioxide” | “residual sugar” |

<img width="1299" alt="截圖 2023-09-17 14 39 04" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/bd999efa-45ff-4408-b1fa-089aaedfcdea">

# Red

**Grid Search CV**

```python
params = {
        'C': np.logspace(-3, 3, num = 7, base = 10),
        'kernel': ['rbf', 'sigmoid'],
        'gamma': np.logspace(-3, 3, num = 7, base = 10),  
    }

Before SMOTE
	Best params: {'C': 10, gamma: 100, kernel: 'rbf'}
	Best scores: 0.6035

After SMOTE
	Best params: {'C': 1000, gamma: 10, kernel: 'rbf'}
	Best scores: 0.8404
```


**Feature Importance**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/9b078489-0591-46bb-9c57-bf963db4687a">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/63af2770-0a08-4c90-8419-84112a6c0fa1">

Fig. 15-1 & Fig. 15-2


**Model Performance**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/544f3fd2-eb75-4087-affb-c1cf0cad0bc9">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/b87bd2bd-b7b6-4972-9a83-358bb57561bf">

Fig. 15-3 & Fig. 15-4

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/c48729eb-cd33-43ee-ba3f-15ef45e324e2">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/1f3c849e-3b57-4622-9d49-177949f7cafa">

Fig. 15-5 & Fig. 15-6

---

# White

**Grid Search CV**

```python
params = {
        'C': np.logspace(-3, 3, num = 7, base = 10),
        'kernel': ['rbf', 'sigmoid'],
        'gamma': np.logspace(-3, 3, num = 7, base = 10),  
    }

Before SMOTE
	Best params: {'C': 1, gamma: 100, kernel: 'rbf'}
	Best scores: 0.6192

After SMOTE
	Best params: {'C': 1000, gamma: 1, kernel: 'rbf'}
	Best scores: 0.8794
```

**Feature Selection**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/3816f0b7-4b9c-4edb-89e4-88366e151304">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/95c6d715-6a7f-42e6-b08b-7037ec014e45">

Fig. 16-1 & Fig. 16-2

**Model Performance**

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/634131ea-6f60-424b-bfc0-a7ff70ee7cf5">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/0b4e28b8-49d9-471c-b3d0-6d9fbb4f48cb">
![newplot](https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/ffccffc2-210c-40e4-99c0-2e19b42bf70f)

Fig. 16-3 & Fig. 16-4

<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/c91fcc5a-12f1-401c-8534-3f4bce68d426">
<img width="450" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/cc631eeb-e8d4-4d08-b3ab-2a67f80806ff">

Fig. 16-5 & Fig. 16-6

# Results and Discussion

最後，我想針對 P. Cortez *et al.* 的文章做一些討論，從其文章中的 Table 2 可以看到，他用來衡量模型的標準是 Accuracy，但正如我的 Fig. 14 顯示，此資料集的 Quality 是集中在 5~6 這個區間，而作者也有畫出他們的 Confusion matrix，可以看到他們並無做 Over-sampling 等處理，表示其 Accuracy 數值會嚴重因為資料集中的問題而提高，我嘗試復刻他們的實驗（沒有轉換, 沒有 Over-sampling）而得到的 Classification Report 大致如下，雖然因為參數設定不同等原因不會完全一樣，但仍可以作為參考：

**Red wine**

<img width="520" alt="截圖 2023-05-06 01 13 18" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/1c0c9b25-56e2-4d0d-bfa1-91f74f170a94">
<img width="260" alt="截圖 2023-05-06 01 27 58" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/4f262b96-1731-4a17-9f21-f6ee0412cb1b">

**White wine**

<img width="520" alt="截圖 2023-05-06 01 13 36" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/4396553d-53a2-455d-ba36-eb3eec497200">
<img width="260" alt="截圖 2023-05-06 01 26 49" src="https://github.com/scfengv/Wine-Type-and-Quality-Classification/assets/123567363/0cf4bb7e-2f76-4bdc-a5ff-d476807c653a">


從上述公式可以將 Precision 和 Recall 理解成，Precision 是計算被分類到此類的所有數據中，真的屬於這個類別的機率，而 Recall 則是所有屬於這個類別的數據中，真的被分到這類的機率。可以在 Red wine 的 Quality = 5 和 White wine 的 Quality =  6 中看到 Low Precision & High Recall 的現象，即表示被分到這個類別的資料分別有 54% 和 46% 不屬於這個類別，但屬於這個類別的資料都有確實被分類到此處。而在 Red wine 的 Quality = 6, 7 和 White wine 的 Quality = 4, 5, 7, 8 都有看到 High Precision & Low Recall 的現象，表示雖然被分到這個類別的資料確實都屬於此處，但該類別仍有很多資料四散在其他類別 (即前面提到的 Low Precision & High Recall 處)。

基於以上理由我認為這篇論文使用 Accuracy 作為衡量模型的標準是不恰當的，而應該是使用綜合評估 Precision 和 Recall 的 F1 score，以處理不平衡資料的分佈問題，而我前面之所以用 Accuracy 來做是因為我有對資料集做 Over-sampling 來使資料集變為平衡，故使用 Accuracy 可以綜合評估分類的正確性。

### Reference:

[1] Paulo Cortez, António Cerdeira, Fernando Almeida, Telmo Matos, José Reis, Modeling wine preferences by data mining from physicochemical properties, Decision Support Systems, Volume 47, Issue 4, 2009

[2] James G, Witten D, Hastie T, Tibshirani R. An Introduction to Statistical Learning: With Applications in R. 1st ed. 2013, Corr. 7th printing 2017 edition. Springer; 2013.

[3] An Introduction to Support Vector Regression [https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2](https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2)

[4] R筆記 – (14)Support Vector Machine/Regression(支持向量機SVM) [https://rpubs.com/skydome20/R-Note14-SVM-SVR](https://rpubs.com/skydome20/R-Note14-SVM-SVR)

[5] 5 SMOTE Techniques for Oversampling your Imbalance Data. [https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5](https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5)

[6] SMOTE API [https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
