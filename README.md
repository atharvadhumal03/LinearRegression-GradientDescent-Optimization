# LinearRegression-GradientDescent-Optimization
Implementation of Simple Linear Regression from scratch using Python to optimize parameters using gradient descent, predict target values, evaluate model performance, and compare against scikit-learn’s regression model.

## 📊 Project Overview
This project demonstrates how to implement Simple Linear Regression (SLR) from scratch, without relying on machine learning libraries like scikit-learn. The focus is on building an interpretable, transparent pipeline using pure Python and applying gradient descent to optimize model parameters. The model is tested on a real-world dataset to predict numerical outcomes and compared against the industry-standard LinearRegression from sklearn.

### Objectives:
- Understand the mathematical intuition behind linear regression.
- Implement gradient descent algorithm manually for optimizing slope and intercept.
- Compare performance (RMSE and R²) between custom implementation and sklearn’s.
- Visualize the regression line and assess model performance.

## 📈 Results Summary
After running the regression using both implementations on the same dataset, the following results were observed:
| **Metric** | **Custom GD Implementation** | **sklearn Implementation** |
|------------|------------------------------|----------------------------|
| **RMSE** | 3882.71 | 5076.83 |
| **R² Score** | 0.98 | 0.94 |
| **Iterations** | 2000 | N/A (analytical solution) |
| **Learning Rate** | 0.01 | N/A |

✅ Observation: With a well-tuned learning rate, adequate stopping at step size and sufficient iterations, the custom implementation outperforms sklearn in terms of RMSE and R² — indicating excellent convergence and prediction accuracy.

## ⚙️ Methodology
### 1. Data Preparation
- Loaded feature (X) and target (y) arrays.
- Converted NumPy arrays to native Python lists to maintain compatibility with manually implemented functions.
- Created a custom train_test_split() function to divide data into 80% training and 20% testing sets.

### 2. Gradient Descent Optimization
Implemented gradient descent manually to minimize the Mean Squared Error (MSE) loss function:
- Loss Function: $$\text{Loss} = \frac{1}{n} \sum_{i=1}^{n}(y_i - (mx_i + c))^2$$
- Parial Derivatives (single data point):
  - with respect to intercept (c) = $$\frac{dL}{dc} = -2(y_i - (mx_i + c))$$
    
  - with respect to slope (m) = $$\frac{dL}{dm} = -2x(y_i - (mx_i + c))$$
- Computing values:
  - intercept $$c = c - \alpha \cdot \frac{dL}{dc}$$
    
  - slope $$m = m - \alpha \cdot \frac{dL}{dm}$$
    
Iterated until both step sizes for slope and intercept became smaller than a defined threshold (0.001) or until max iterations (2000) were reached.

### 4. Comparison with sklearn
- Trained a model using LinearRegression() from scikit-learn.
- Compared results to validate the custom implementation.
- Demonstrated how gradient descent can be tuned to match or even outperform sklearn’s analytical solution in some cases.

## 📊 Visualization
- Plotted y_test vs x_test as a scatter plot.
- Overlaid the predicted regression line using y_pred values to visually validate the model.
Visual inspection confirmed that the regression line fits the data well, with minimal residuals.
