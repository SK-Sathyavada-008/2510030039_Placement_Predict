import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from src.data.ingest import load_and_validate_data


def compute_cost(X, y, w):
    """
    Computes Mean Squared Error (MSE) cost function.
    """

    m = len(y)

    predictions = np.dot(X, w)

    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


def gradient_descent(
    X,
    y,
    w,
    alpha,
    num_iters
):
    """
    Implements Gradient Descent optimization
    from scratch.
    """

    m = len(y)

    cost_history = []

    for i in range(num_iters):

        predictions = np.dot(X, w)

        errors = predictions - y

        # Gradient
        gradient = (
            (1 / m)
            * np.dot(X.T, errors)
        )

        # Update weights
        w = w - alpha * gradient

        # Track cost
        cost = compute_cost(
            X,
            y,
            w
        )

        cost_history.append(cost)

    return w, cost_history


def run_gradient_descent_experiment():

    # 1. Load data

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(
        DATA_PATH
    )

    # Select feature and target

    feature_cols = ['cgpa']

    target_col = 'salary_package_lpa'

    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X_raw = df_clean[
        feature_cols
    ].values

    y_raw = df_clean[
        target_col
    ].values.reshape(-1, 1)

    # 2. 80/20 train-test split

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_raw,
            y_raw,
            test_size=0.20,
            random_state=42
        )
    )

    # Feature scaling

    scaler_x = StandardScaler()

    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(
        X_train
    )

    X_test_scaled = scaler_x.transform(
        X_test
    )

    y_train_scaled = scaler_y.fit_transform(
        y_train
    )

    # Add intercept column

    X_train_design = np.hstack(
        [
            np.ones(
                (
                    X_train_scaled.shape[0],
                    1
                )
            ),
            X_train_scaled
        ]
    )

    # 3. Learning rate experiments

    learning_rates = [
        0.001,
        0.01,
        0.1,
        0.5
    ]

    num_iterations = 1000

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    results = {}

    for alpha in learning_rates:

        # Initialize weights

        w_init = np.zeros(
            (
                X_train_design.shape[1],
                1
            )
        )

        # Run gradient descent

        w_opt, cost_history = (
            gradient_descent(
                X_train_design,
                y_train_scaled,
                w_init,
                alpha,
                num_iterations
            )
        )

        results[alpha] = {
            'weights': w_opt,
            'history': cost_history
        }

        # Plot cost

        plt.plot(
            cost_history,
            label=f"Alpha (α) = {alpha}"
        )

    plt.xlabel(
        "Iterations"
    )

    plt.ylabel(
        "Cost Function E(w) - MSE"
    )

    plt.title(
        "Effect of Different Learning Rates "
        "on Gradient Descent Convergence"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "reports/figures/"
        "gd_learning_rates_comparison.png"
    )

    plt.close()

    print(
        "-> Saved learning rates cost graph to "
        "reports/figures/"
        "gd_learning_rates_comparison.png"
    )

    # Select optimal alpha

    best_alpha = 0.1

    final_w = results[
        best_alpha
    ]['weights']

    print("\n" + "=" * 50)

    print(
        f"--- CUSTOM GRADIENT DESCENT "
        f"PARAMETERS (α = {best_alpha}) ---"
    )

    print(
        f"Intercept (w0): "
        f"{final_w[0, 0]:.4f}"
    )

    print(
        f"Coefficient (w1): "
        f"{final_w[1, 0]:.4f}"
    )

    # 4. Compare with Scikit-learn

    sklearn_model = (
        SklearnLinearRegression()
    )

    sklearn_model.fit(
        X_train_scaled,
        y_train_scaled
    )

    print("\n" + "=" * 50)

    print(
        "--- SCIKIT-LEARN COMPARISON ---"
    )

    print(
        f"Scikit-learn Intercept: "
        f"{sklearn_model.intercept_[0]:.4f}"
    )

    print(
        f"Scikit-learn Coefficient: "
        f"{sklearn_model.coef_[0, 0]:.4f}"
    )


if __name__ == "__main__":
    run_gradient_descent_experiment()