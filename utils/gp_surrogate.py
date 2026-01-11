from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float, sigma_f: float) -> np.ndarray:
    """
    Compute the Squared Exponential (RBF) covariance matrix between X1 and X2.

    Formula:
        k(x, x') = sigma_f^2 * exp( -0.5 * ||x - x'||^2 / l^2 )

    Args:
        X1 (np.ndarray): shape (n1, D)
        X2 (np.ndarray): shape (n2, D)
        length_scale (float): The length scale 'l', controlling smoothness.
        sigma_f (float): The signal variance 'sigma_f', controlling amplitude.

    Returns:
        np.ndarray: Covariance matrix of shape (n1, n2).
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    # ||x - x'||^2 = (x^2)_i + (x'^2)_j - 2 x_i·x'_j
    sq_norms1 = np.sum(X1**2, axis=1)[:, None]
    sq_norms2 = np.sum(X2**2, axis=1)[None, :]
    sq_dists = sq_norms1 + sq_norms2 - 2.0 * X1 @ X2.T

    return (sigma_f**2) * np.exp(-0.5 * sq_dists / (length_scale**2))


@dataclass
class GaussianProcessSurrogate:
    """
    A lightweight Gaussian Process Regressor using an RBF Kernel.

    Designed for the COD Assignment to avoid heavy dependencies (like sklearn).
    
    Attributes:
        length_scale (float): Kernel length scale (fixed).
        sigma_f (float): Kernel signal variance (fixed).
        sigma_n (float): Observation noise variance (for numerical stability).
    """
    length_scale: float = 0.3
    sigma_f: float = 1.0
    sigma_n: float = 1e-6  # observation noise

    # Internal attributes (filled by fit)
    X_train_: Optional[np.ndarray] = None
    y_train_: Optional[np.ndarray] = None
    X_min_: Optional[np.ndarray] = None
    X_max_: Optional[np.ndarray] = None
    y_mean_: Optional[float] = None
    y_std_: Optional[float] = None
    L_: Optional[np.ndarray] = None          # Cholesky of K
    alpha_: Optional[np.ndarray] = None      # (K^-1 y) vector

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalize input X to [0, 1] using training bounds."""
        X = np.asarray(X, float)
        return (X - self.X_min_) / (self.X_max_ - self.X_min_ + 1e-12)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        """
        Train the GP model on the provided dataset.

        Performs:
        1.  **Normalization:** Scales X to [0, 1] and y to Standard Normal (0, 1).
        2.  **Kernel Construction:** Builds the covariance matrix K.
        3.  **Cholesky Decomposition:** Computes L such that K = L @ L.T.
        4.  **Weights:** Solves for alpha = K^-1 * y.

        Args:
            X (np.ndarray): Input training data (n_samples, n_features).
            y (np.ndarray): Target values (n_samples,).

        Returns:
            self: The fitted regressor.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        # Store scaling for X
        self.X_min_ = X.min(axis=0)
        self.X_max_ = X.max(axis=0)
        X_scaled = self._scale_X(X)

        # Standardise y
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std() if y.std() > 0 else 1.0)
        y_std = (y - self.y_mean_) / self.y_std_

        # Kernel matrix + noise
        K = _rbf_kernel(X_scaled, X_scaled, self.length_scale, self.sigma_f)
        K[np.diag_indices_from(K)] += self.sigma_n**2

        # Cholesky factorisation
        self.L_ = np.linalg.cholesky(K)
        # Solve for alpha = K^-1 y_std via L
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, y_std))

        self.X_train_ = X_scaled
        self.y_train_ = y_std
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict the mean and standard deviation for new inputs.

        Args:
            X (np.ndarray): New input points (n_samples, n_features).
            return_std (bool): If True, returns the predictive standard deviation (uncertainty).

        Returns:
            y_mean (np.ndarray): Predicted values (rescaled to original units).
            y_std (np.ndarray or None): Uncertainty estimates (if return_std=True).
        """
        if self.X_train_ is None:
            raise RuntimeError("GP not fitted yet. Call fit() first.")

        X = np.asarray(X, float)
        X_scaled = self._scale_X(X)

        # Cross-kernel between training and test
        K_star = _rbf_kernel(self.X_train_, X_scaled, self.length_scale, self.sigma_f)
        # Predictive mean in standardised space
        y_mean_std = K_star.T @ self.alpha_

        # Rescale back to original y units
        y_mean = y_mean_std * self.y_std_ + self.y_mean_

        if not return_std:
            return y_mean, None

        # Solve v = L^-1 K_star
        v = np.linalg.solve(self.L_, K_star)
        # Predictive variance in std space: k(x*,x*) - v^T v
        k_xx = (self.sigma_f**2) * np.ones(X_scaled.shape[0])
        y_var_std = k_xx - np.sum(v**2, axis=0)
        y_var_std = np.maximum(y_var_std, 1e-12)  # clamp numerical noise

        # Rescale variance: var(y) = (std_y^2) * var(std_y)
        y_std = np.sqrt(y_var_std) * self.y_std_
        return y_mean, y_std
