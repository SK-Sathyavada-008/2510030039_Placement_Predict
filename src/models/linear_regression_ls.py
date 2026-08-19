import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.data.ingest import load_and_validate_data


def train_linear_regression_ls():

    # 1. Load Data
    DATA_PATH = os.path.join(
        "src", "data", "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)

    # Remove rows with missing required values
    df_clean = df.dropna(
        subset=[
            'cgpa',
            'communication_skills',
            'salary_package_lpa'
        ]
    ).copy()

    # 2. Define input and output
    feature_cols = [
        'cgpa',
        'communication_skills'
    ]

    target_col = 'salary_package_lpa'

    X_raw = df_clean[feature_cols].values

    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]

    print(
        f"Loaded {N} data points with "
        f"input dimension L = {X_raw.shape[1]} "
        f"and output dimension M = {y.shape[1]}"
    )

    # 3. Create design matrix
    X_design = np.hstack(
        [
            np.ones((N, 1)),
            X_raw
        ]
    )

    # 4. Normal Equation
    # w = (X^T X)^-1 X^T y

    XT_X = np.dot(
        X_design.T,
        X_design
    )

    try:

        XT_X_inv = np.linalg.inv(XT_X)

    except np.linalg.LinAlgError:

        # Use pseudo-inverse if matrix is singular
        XT_X_inv = np.linalg.pinv(XT_X)

    XT_y = np.dot(
        X_design.T,
        y
    )

    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )

    print(
        "\n--- Optimal Model Parameters "
        "(Weights & Bias) ---"
    )

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.4f}"
    )

    print(
        f"Coefficient for {feature_cols[0]} "
        f"(w1): {w_optimal[1, 0]:.4f}"
    )

    print(
        f"Coefficient for {feature_cols[1]} "
        f"(w2): {w_optimal[2, 0]:.4f}"
    )

    # Calculate predictions
    y_pred = np.dot(
        X_design,
        w_optimal
    )

    # Calculate error
    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )

    print(
        f"Minimized Error (E_w): "
        f"{E_w:.4f}"
    )

    # 5. 3D Visualization

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        projection='3d'
    )

    # Actual data points

    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.ravel(),
        color='blue',
        alpha=0.6,
        label='Actual Data Points'
    )

    # Create regression plane

    x1_surf = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        20
    )

    x2_surf = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        20
    )

    x1_mesh, x2_mesh = np.meshgrid(
        x1_surf,
        x2_surf
    )

    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        color='red',
        alpha=0.3,
        edgecolor='none'
    )

    ax.set_xlabel(
        'CGPA (Feature 1)'
    )

    ax.set_ylabel(
        'Communication Skills (Feature 2)'
    )

    ax.set_zlabel(
        'Salary Package LPA (Target)'
    )

    ax.set_title(
        'Linear Regression via Standard Least Squares '
        '(P=1, L=2, M=1)'
    )

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )

    plt.savefig(output_path)

    plt.close()

    print(
        f"\n-> Successfully saved 3D "
        f"regression plot to {output_path}"
    )


if __name__ == "__main__":
    train_linear_regression_ls()