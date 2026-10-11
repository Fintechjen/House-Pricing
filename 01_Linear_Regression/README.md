# Linear Regression From Scratch

## Overview

This project implements **Linear Regression from scratch using NumPy and Gradient Descent** on the Kaggle **House Prices - Advanced Regression Techniques** dataset.

The goal of this experiment is to understand the complete linear regression pipeline from first principles rather than directly using a machine-learning library.

### What is implemented?

- Linear regression mathematically
- Mean Squared Error (MSE)
- Gradient descent
- Matrix-based implementation
- Manual missing-value handling
- Manual one-hot encoding
- Feature standardization
- Log transformation of the target
- RMSLE evaluation
- Learning-rate experiments
- Epoch/convergence experiments
- Prediction and error analysis
- Kaggle submission

---

## 1. Mathematical Foundation

The linear regression model is:

$$
\hat{y} = X\beta
$$

where:

- $X$ is the feature matrix
- $\beta$ is the parameter vector
- $\hat{y}$ is the prediction

The model minimizes Mean Squared Error:

$$
J(\beta)
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
$$

Since:

$$
\hat{y}=X\beta
$$

the gradient of the cost function is:

$$
\nabla J(\beta)
=
\frac{2}{n}
X^T(X\beta-y)
$$

Gradient descent updates the parameters using:

$$
\beta
\leftarrow
\beta-\alpha\nabla J(\beta)
$$

where $\alpha$ is the learning rate.

The intercept is implemented by adding a column of ones to the feature matrix:

$$
X=
\begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \cdots \\
1 & x_1^{(2)} & x_2^{(2)} & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

---

## 2. Gradient Descent Implementation

The core model was implemented without using a regression estimator from `scikit-learn`.

```python
class LinearRegressionGD:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def fit(self, X, y):

        X = np.column_stack([
            np.ones(X.shape[0]),
            X
        ])

        n, p = X.shape

        self.beta = np.zeros(p)

        for epoch in range(self.epochs):

            predictions = X @ self.beta

            errors = predictions - y

            cost = np.mean(errors ** 2)

            self.loss_history.append(cost)

            gradient = (2 / n) * X.T @ errors

            self.beta -= self.learning_rate * gradient

        return self

    def predict(self, X):

        X = np.column_stack([
            np.ones(X.shape[0]),
            X
        ])

        return X @ self.beta
```

This implementation uses **full-batch gradient descent**. Each update uses the complete training dataset.

---

## 3. Data Preprocessing

The House Prices dataset contains both numerical and categorical variables.

### Missing values

Missing values were handled according to the meaning of the feature.

Examples:

- Garage/basement categorical absence → `"None"`
- Garage/basement numerical absence → `0`
- Masonry veneer absence → `"None"` / `0`
- `LotFrontage` → training-set median
- `Electrical` → training-set mode

The imputation values are learned from the training data and then applied to validation data.

### Categorical variables

Categorical features were manually converted into binary indicator variables.

For each categorical feature, one category was used as the baseline and the remaining categories were represented using dummy variables.

This allows categorical information to be incorporated into:

$$
X\beta
$$

without assigning arbitrary numerical values to categories.

### Feature scaling

Features were standardized using:

$$
z=
\frac{x-\mu}{\sigma}
$$

where $\mu$ and $\sigma$ were calculated from the training set.

The same values were then used to transform the validation set.

Scaling is particularly important for gradient descent because features with very different magnitudes can lead to inefficient optimization.

---

## 4. Target Transformation

The competition evaluates predictions using RMSLE.

Therefore, the target was transformed into log space:

```python
y_log = np.log1p(y)
```

The model therefore learns:

$$
\log(1+y)
$$

rather than the raw sale price.

Predictions are converted back using:

```python
np.expm1(prediction)
```

The RMSLE metric is:

$$
RMSLE
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left[
\log(1+\hat{y}_i)
-
\log(1+y_i)
\right]^2
}
$$

---

## 5. Baseline Results

The initial model used:

```text
Learning rate = 0.01
Epochs = 5000
```

Results:

| Metric | Result |
|---|---:|
| Training RMSLE | 0.09174 |
| Validation RMSLE | 0.12506 |
| Validation RMSE | $22,143.74 |

The optimized configuration tested later achieved approximately:

$$
\boxed{\text{Validation RMSLE}=0.1249}
$$

---

## 6. Learning Rate Experiment

The learning rate controls the size of each gradient descent step:

$$
\beta_{\text{new}}
=
\beta_{\text{old}}
-
\alpha\nabla J
$$

The following learning rates were tested:

| Learning Rate | Train RMSLE | Validation RMSLE |
|---:|---:|---:|
| 0.001 | 0.092926 | 0.126525 |
| 0.003 | 0.091862 | 0.125088 |
| 0.010 | 0.091741 | 0.125064 |
| **0.030** | **0.091729** | **0.124905** |
| 0.100 | NaN | NaN |

### Observation

A learning rate that is too small leads to slow convergence.

A moderate learning rate converges faster and achieved the best validation result among the tested values.

A learning rate of `0.1` became unstable and produced `NaN`, demonstrating that excessively large updates can cause gradient descent to diverge.

The best tested learning rate was:

$$
\boxed{\alpha=0.03}
$$

---

## 7. Epoch Experiment

An epoch corresponds to one full-batch gradient descent update in this implementation.

| Epochs | Train RMSLE | Validation RMSLE |
|---:|---:|---:|
| 100 | 1.598635 | 1.595929 |
| 500 | 0.092924 | 0.126527 |
| 1,000 | 0.092063 | 0.125295 |
| 2,000 | 0.091796 | 0.125078 |
| 5,000 | 0.091741 | 0.125064 |
| 10,000 | 0.091733 | 0.124963 |

### Observation

The largest improvement occurs during the early optimization stages.

After approximately 2,000–5,000 epochs, additional epochs provide only marginal improvements.

This demonstrates the difference between:

$$
\text{Learning Rate}=\text{Step Size}
$$

and

$$
\text{Epochs}=\text{Number of Updates}
$$

---

## 8. Error Analysis

Errors were analyzed in both dollar space and log space.

A large dollar error does not necessarily correspond to a large RMSLE error.

For example, an error of \$25,000 on a \$40,000 house represents a much larger relative difference than a \$100,000 error on a \$600,000 house.

Because:

$$
\log(1+\hat y)-\log(1+y)
$$

is related to the ratio between predicted and actual prices, RMSLE places greater emphasis on proportional errors.

This is why the ranking of errors in dollar space differs from the ranking in log space.

---

## 9. Kaggle Result

The final model was retrained using the complete training dataset and used to generate predictions for the Kaggle test set.

The submission achieved:

$$
\boxed{\text{Kaggle RMSLE}=0.15099}
$$

This was higher than the local validation score:

$$
0.1249
\rightarrow
0.15099
$$

The difference became an additional experiment to investigate.

Potential explanations being investigated include:

- train/test distribution differences
- differences between a single validation split and the hidden Kaggle test set
- preprocessing differences
- generalization limitations of linear regression

No unseen categorical levels were found when comparing the training and test categorical features.

---

## 10. Key Learnings

This experiment established the complete linear regression workflow:

$$
\boxed{
\text{Data}
\rightarrow
\text{Preprocessing}
\rightarrow
\text{Feature Matrix}
\rightarrow
\text{Gradient Descent}
\rightarrow
\text{Prediction}
\rightarrow
\text{Evaluation}
}
$$

More importantly, the model was implemented from the underlying mathematics rather than treating linear regression as a black box.

The next experiment will compare this implementation against a library implementation of linear regression using the same data and evaluation methodology.

---

## Next Experiment

**Linear Regression with Libraries**

The next experiment will use `scikit-learn` to implement linear regression and compare it against this from-scratch implementation.

The comparison will use:

- the same features
- the same preprocessing
- the same train/validation split
- the same log-transformed target
- the same RMSLE metric

The objective is to determine whether the remaining error comes from the implementation/optimization or from the limitations of the linear regression model itself.
