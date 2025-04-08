# Classification Using Decision Trees, AdaBoost+Decision Trees, and SVM

Boyu Yue

ybyleo@126.com

4/8/25

## Introduction

In this assignment, I will use Decision Trees, AdaBoost+Decision Trees, and SVM (with three different kernels) to classify the given 3D-make-moons data, and compare their classification result. 

##  Decision Trees

A decision tree is a hierarchical model that recursively splits the input feature space into regions based on feature values. Each internal node represents a decision based on a feature, each branch represents the outcome of that decision, and each leaf node corresponds to a class label. 

### Methodology

Let $X={x1,x2,...,xn} $ represent the input dataset, where each $x_i$ is a feature vector with $m$ features,  $x_i=(x_{i1}, x_{i2}, ..., x_{im})$, and $y_i \in \{C_1, C_2, ..., C_k\}$ is the corresponding class label from $k$ possible classes. The decision tree partitions the feature space into disjoint regions $R_1, R_2, ..., R_p$, where each region $R_j$ is associated with a dominant class label.

At each node, the tree selects a feature and a threshold to split the data into two subsets. For a feature  $x_j$ and threshold $t$, the split divides the data into:

Left subset: ${xi∣xij\leq t}$

Right subset: ${xi∣xij>t}$

The quality of the split is evaluated using a criterion that measures the impurity of the resulting subsets. Common impurity measures for classification include  Gini Index and Entropy (Information Gain). 

#### Gini Index

The Gini Index measures the impurity of a node by calculating the probability of incorrectly classifying a randomly chosen sample if it were labeled according to the distribution of classes in that node.

For a node with class distribution $p_1, p_2, ..., p_k$ (where $p_j$ s the proportion of class $C_j$), the Gini Index is:
$$
Gini = 1 - \sum_{j=1}^{k} p_j^2
$$
For a split into left ($L$) and right ($R$) subsets with $N_L$ and $N_R$ samples respectively (total samples $N = N_L + N_R$), the weighted Gini Index is:
$$
Gini_{\text{split}} = \frac{N_L}{N} Gini(L) + \frac{N_R}{N} Gini(R)
$$
The algorithm selects the feature and threshold that minimize $Gini_{\text{split}}$.

#### Entropy and Information Gain

Entropy measures the uncertainty in a node’s class distribution, where $p_j$ is the proportion of class $C_j$.:
$$
Entropy = -\sum_{j=1}^{k} p_j \log_2(p_j)
$$
Information Gain quantifies the reduction in entropy after a split:
$$
 IG = \text{Entropy}(\text{parent}) - \left( \frac{N_L}{N} \text{Entropy}(L) + \frac{N_R}{N} \text{Entropy}(R) \right) 
$$
The decision tree chooses the split that maximizes $IG$.

The decision rule at leaf nodes is: once splitting stops, each leaf node is assigned a class label. The label is typically the majority class in that region.

### Experimental Studies

The accuracy of the Decision Tree model is 0.9460, with the result show as below:

<img src="../../../HuaweiMoveData/Users/岳伯禹/Documents/WeChat Files/wxid_7a4ye1q6cjvf32/FileStorage/Temp/1744079518725.png" alt="1744079518725" style="zoom:50%;" />

## AdaBoost + Decision Trees

Adaptive Boosting (AdaBoost) is an ensemble learning method that combines multiple weak learners to create a strong classifier. It iteratively adjusts the importance of training samples and weights the contribution of each weak learner to improve overall accuracy. When paired with decision trees, AdaBoost leverages their splitting mechanism while introducing a boosting framework.

### Methodology

First start with equal weights for all samples, where $w_i^{(t)}$ is the weight of sample $x_i$ at iteration $t$:
$$
w_i^{(1)} = \frac{1}{n}, \quad i = 1, 2, \dots, n 
$$
Fit a decision tree $h_t(x)$ to the weighted dataset. The tree uses a splitting criterion, but minimizes the weighted error:
$$
e_t = \sum_{i=1}^{n} w_i^{(t)} I(y_i \neq h_t(x_i))
$$
Calculate the weight $\alpha_t$ of the classifier $h_t$, based on its error. $\alpha_t$ measures the classifier’s importance, a lower error yields a higher $\alpha_t$. When error equals to 0.5, the $\alpha_t$ is 0:
$$
\alpha_t = \frac{1}{2} \ln \left( \frac{1 - e_t}{e_t} \right)
$$
Update sample weights, Adjust the weights to focus on misclassified samples and normalize the weights so they sum to 1:
$$
w_i^{(t+1)} = w_i^{(t)} \exp \left( -\alpha_t y_i h_t(x_i) \right)
$$

$$
w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^{n} w_j^{(t+1)}}
$$

Combine the $T$ weak classifiers into a strong classifier:
$$
H(x) = \text{sign} \left( \sum_{t=1}^{T} \alpha_t h_t(x) \right)
$$
The output $H(x) \in \{-1, 1\}$ is the weighted majority vote of all decision trees.

### Experimental Studies

The accuracy of AdaBoost + Decision Tree is 0.9660, with the result show as below:

<img src="../../../HuaweiMoveData/Users/岳伯禹/Documents/WeChat Files/wxid_7a4ye1q6cjvf32/FileStorage/Temp/1744081942256.png" alt="1744081942256" style="zoom:50%;" />

## SVM

Support Vector Machines (SVM) are supervised learning models used for classification by finding the optimal hyperplane that maximally separates classes in the feature space. For non-linearly separable data, SVM employs kernel functions to transform the data into a higher-dimensional space where a linear boundary can be established.

### Methodology

SVM aims to find a hyperplane defined by:
$$
 w^T x + b = 0
$$
The goal is to maximize the margin—the distance between the hyperplane and the nearest data point from either class.

For margin maximization, the optimization problem is:
$$
\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w^T x_i + b) \geq 1, \quad \forall i
$$
The constraint ensures that all points lie outside the margin, with $y_i (w^T x_i + b)$ being the functional margin. To solve this constrained problem, SVM uses Lagrange multipliers $\alpha_i \geq 0$:
$$
L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i [y_i(w^T x_i + b) - 1]
$$
Maximizing the dual form:
$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j
$$
For a new input $x$, the prediction:
$$
f(x) = \text{sign} \left( \sum_{i=1}^{n} \alpha_i y_i x_i^T x + b \right)
$$
For non-linear boundaries, SVM maps data into a higher-dimensional space via a function $\phi(x)$, replacing $x_i^T x_j$ with a kernel function $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$. This avoids explicitly computing $\phi(x)$. The dual becomes:
$$
\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(x_i, x_j)
$$

#### Linear Kernel

Definition:
$$
K(x_i, x_j) = x_i^T x_j
$$
There's no transformation.It assumes data is linearly separable in the original space.

#### Polynomial Kernel

Definition:
$$
K(x_i, x_j) = (x_i^T x_j + r)^d 
$$
Where $d$ is the degree of the polynomial, $r$ is the offset, which controls the influence of higher-order terms.

#### Radial Basis Function (RBF) Kernel

Definition:
$$
K(x_i, x_j) = \exp \left( -\gamma \|x_i - x_j\|^2 \right)
$$
where $\gamma = \frac{1}{2\sigma^2}$ controls the width of the Gaussian. Smaller $\gamma$ means a wider influence.

Prediction with kernels become:

### Experimental Studies

The accuracy of linear kernel is 0.6680:

<img src="../../../HuaweiMoveData/Users/岳伯禹/Documents/WeChat Files/wxid_7a4ye1q6cjvf32/FileStorage/Temp/1744083876717.png" alt="1744083876717" style="zoom:50%;" />

The accuracy of polynomial kernel is 0.7600:

<img src="../../../HuaweiMoveData/Users/岳伯禹/Documents/WeChat Files/wxid_7a4ye1q6cjvf32/FileStorage/Temp/1744083927231.png" alt="1744083927231" style="zoom:50%;" />

The accuracy of rbf kernel is 0.9680:

<img src="../../../HuaweiMoveData/Users/岳伯禹/Documents/WeChat Files/wxid_7a4ye1q6cjvf32/FileStorage/Temp/1744083962321.png" alt="1744083962321" style="zoom:50%;" />

## Comparison and Conclusion

For the classification task, using decision trees can achieve a great classification result. Adding adaboost improves the results of decision trees. However, since decision trees can achieve high accuracy by themselves, the improvement space of adaboost is limited and not obvious.

For SVM with different kernels, since the shape of the data itself is highly nonlinear, the prediction effect of the linear kernel is average. The poly kernel can be suitable for some linear data, and some improvements have been made on this basis. Rbf kernel is very suitable for nonlinear data classification and has a better effect, which achieved the best accuracy in the whole assignment.