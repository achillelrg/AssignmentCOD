from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import warnings

# Try to import scikit-learn for Gaussian Process
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    warnings.warn("scikit-learn not found. Surrogate model will use a simple RBF interpolation fallback (scipy).")

# Try to import scipy for fallback
try:
    from scipy.interpolate import Rbf
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class SurrogateModel:
    """
    A wrapper for a surrogate model (Kriging/Gaussian Process or RBF).
    Used for Part C of the assignment.
    """
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the surrogate model on data (X, y).
        X: (N, D) array
        y: (N, ) array
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

        if _HAS_SKLEARN:
            # Kriging (Gaussian Process)
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
            self.model.fit(self.X_train, self.y_train)
        elif _HAS_SCIPY:
            # Simple RBF Interpolation fallback
            # smooth=0.1 to avoid strict interpolation (noise handling)
            self.model = Rbf(*self.X_train.T, self.y_train, function='multiquadric', smooth=0.1)
        else:
            raise RuntimeError("No suitable library (scikit-learn or scipy) found for surrogate modelling.")
        
        self.is_fitted = True

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict values.
        If return_std is True, also return uncertainty (sigma) if supported.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        X = np.asarray(X)
        # Ensure 2D
        if X.ndim == 1:
            X = X[np.newaxis, :]

        if _HAS_SKLEARN:
            return self.model.predict(X, return_std=return_std)
        elif _HAS_SCIPY:
            # Scipy Rbf doesn't give std directly
            pred = self.model(*X.T)
            if return_std:
                return pred, np.zeros_like(pred) # Dummy 0 std
            return pred, None
        return np.zeros(len(X)), None 
