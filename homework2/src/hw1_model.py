"""
Homework 1: Regression and Classification Models

Stencil for:
1. Linear Regression (for genomic methylation data)
2. Logistic Regression (for heart disease data)

Students will implement model training, evaluation, and K-fold cross-validation.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class GenomicAgeRegressor:
    def __init__(
        self,
        model_type: str = "linear",
        random_state: int = 42,
        selected_features: Optional[List[str]] = None,
        alpha: float = 1.0,
    ):
        """
        Initialize the regressor with specified parameters.

        Args:
            model_type: Type of regression model ('linear', 'ridge', 'lasso')
            random_state: Random seed for reproducibility
            selected_features: Optional list of feature names to use
            alpha: Regularization strength for ridge/lasso
        """
        self.model_type = model_type
        self.random_state = random_state
        self.selected_features = selected_features
        self.alpha = alpha

        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Select features and fit the regression model. Save the model as self.model.
        """
        # TODO: Implement model fitting
        # 1. Select features if specified (self.selected_features)
        # 2. Scale features using self.scaler
        # 3. Initialize the model based on self.model_type:
        #       - "linear" -> LinearRegression()
        #       - "ridge"  -> Ridge(alpha=self.alpha, random_state=self.random_state)
        #       - "lasso"  -> Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=10000)
        #    If an unknown model_type is provided, raise a ValueError.
        # 4. Fit the model and store in self.model
        
        X_selected = X
        if self.selected_features is not None:
            X_selected = X[self.selected_features]
        X_scaled = self.scaler.fit_transform(X_selected)

        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "ridge":
            self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.model_type == "lasso":
            self.model = Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=10000)
        else:
            raise ValueError("Unknown model type provided")
        
        self.model.fit(X_scaled, y)
    

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        # TODO: Implement prediction
        # 1. Ensure self.model is trained (raise ValueError if not)
        # 2. Select the same features as used in training (self.selected_features)
        # 3. Scale using self.scaler.transform
        # 4. Return predictions from self.model
        
        if self.model is None:
            raise ValueError("Model is not trained")
        X_selected = X
        if self.selected_features is not None:
            X_selected = X[self.selected_features]
        X_scaled = self.scaler.transform(X_selected)
        return self.model.predict(X_scaled)


    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance and return metrics.
        """
        # TODO: Implement model evaluation
        # Use self.predict(X) then compute:
        #   - mse: mean_squared_error(y, y_pred)
        #   - rmse: sqrt(mse)
        #   - r2:  r2_score(y, y_pred)
        #   - mae: mean_absolute_error(y, y_pred)
        # Return a dict with keys: "mse", "rmse", "r2", "mae"
        
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        eval_dict = {"mse" : mse, "rmse" : rmse, "r2" : r2, "mae" : mae}
        return eval_dict


    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform K-fold cross-validation and return metrics for n_splits folds.

        Returns:
            cv_results: Dictionary with lists of metrics for each fold
        """
        # TODO: Implement K-fold cross-validation using KFold
        # For each fold:
        #   1. Split data into train/val
        #   2. Fit on train
        #   3. Evaluate on val
        #   4. Append metrics to cv_results

        cv_results = {"mse": [], "rmse": [], "r2": [], "mae": []}
        kfold = KFold(n_splits, shuffle=True, random_state=self.random_state)
        
        for train_indices, test_indices in kfold.split(X):
            X_train = X.iloc[train_indices]
            X_val = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_val = y.iloc[test_indices]
            curr_model = GenomicAgeRegressor(model_type=self.model_type, 
                                             random_state=self.random_state, 
                                             selected_features=self.selected_features, 
                                             alpha=self.alpha)
            curr_model.fit(X_train, y_train)
            metrics_dict = curr_model.evaluate(X_val, y_val)
            for metric in cv_results.keys():
                cv_results[metric].append(metrics_dict[metric])
        return cv_results
            
        
class HeartDiseaseClassifier:
    def __init__(
        self,
        C: float = 1.0,
        random_state: int = 42,
        selected_features: Optional[List[str]] = None,
        model_type: str = "linear"
    ):
        """
        Initialize the classifier with specified parameters. Uses the lbfgs solver.

        Args:
            C: Inverse of regularization strength
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.selected_features = selected_features
        self.model_type = model_type

    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for the model.

        Returns:
            X_processed: Processed feature matrix
        """
        # TODO: Implement feature preprocessing
        # Default behavior should return X unchanged
        X_numeric = X.select_dtypes(include="number")
        X_imp = X_numeric.fillna(X_numeric.median(), inplace=True)
        X_selected = X_imp
        if self.selected_features is not None:
            X_selected = X_imp[self.selected_features]
        return X_selected

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Preprocess features and fit the classification model.
        Save the fitted model in self.model.

        Args:
            X: Feature matrix
            y: Target variable (heart disease presence)
        """
        # TODO: Implement model fitting
        # 1. Preprocess features via self.preprocess_features
        # 2. Scale features using self.scaler
        # 3. Initialize LogisticRegression
        # 4. Fit the model and store in self.model
        
        X_preprocessed = self.preprocess_features(X)
        X_scaled = self.scaler.fit_transform(X_preprocessed)
        if self.model_type == "linear":
            self.model = LogisticRegression(C=self.C, random_state=self.random_state)
        elif self.model_type == "lasso":
            self.model = LogisticRegression(C=self.C, 
                                            random_state=self.random_state,
                                            penalty="l1",
                                            solver="saga")
        self.model.fit(X_scaled, y)


    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Make binary predictions using the trained model (self.model).

        Args:
            X: Feature matrix
            return_proba: If True, return probability of class 1 instead of hard labels

        Returns:
            y_pred: Binary predictions (0 or 1) if return_proba=False
            y_proba: Probability predictions for class 1 if return_proba=True
        """
        # TODO: Implement prediction
        # 1. Ensure self.model is trained
        # 2. Preprocess features
        # 3. Scale using self.scaler.transform
        # 4. If return_proba: return self.model.predict_proba(X_scaled)[:, 1]
        #    else: return self.model.predict(X_scaled)
        
        if self.model is None:
            raise ValueError("Model is not trained")
        X_preprocessed = self.preprocess_features(X)
        X_scaled = self.scaler.transform(X_preprocessed)
        if return_proba:
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # TODO: Implement model evaluation
        # Compute:
        #   y_pred  = self.predict(X)
        #   y_proba = self.predict(X, return_proba=True)
        #
        # Use zero_division=0 to avoid crashes on edge cases where a fold has no predicted positives:
        #   precision_score(y, y_pred, zero_division=0)
        #   recall_score(y, y_pred, zero_division=0)
        #   f1_score(y, y_pred, zero_division=0)
        #
        # ROC-AUC safeguard:
        #   Only compute roc_auc_score(y, y_proba) if BOTH classes appear in y (i.e., len(np.unique(y)) == 2).
        #   Otherwise, set "auc" to np.nan (or your chosen sentinel).
        #
        # Return dict with keys: "accuracy", "precision", "recall", "f1", "auc"

        y_pred  = self.predict(X)
        y_proba = self.predict(X, return_proba=True)
        precision_out = precision_score(y, y_pred, zero_division=0, average="weighted")
        recall_out = recall_score(y, y_pred, zero_division=0, average="weighted")
        f1_out = f1_score(y, y_pred, zero_division=0, average="weighted")
        acc_out = accuracy_score(y, y_pred)
        if len(np.unique(y)) > 2:
            auc_out = roc_auc_score(y, y_proba, average="weighted", multi_class="ovr")
        else:
            auc_out = np.nan
        eval_dict = {"accuracy": acc_out, 
                     "precision": precision_out, 
                     "recall": recall_out,
                     "f1": f1_out,
                     "auc": auc_out}
        return eval_dict


    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform K-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of folds for cross-validation

        Returns:
            cv_results: Dictionary with lists of metrics for each fold
        """
        # TODO: Implement K-fold cross-validation using KFold
        # For each fold:
        #   1. Split data into train/val
        #   2. Fit on train
        #   3. Evaluate on val
        #   4. Append metrics to cv_results
        #
        # Note: The ROC-AUC should effectively apply per fold:
        #   If y_val contains only one class, "auc" for that fold should be np.nan (or your chosen sentinel).
        
        cv_results = {"accuracy": [], 
                     "precision": [], 
                     "recall": [],
                     "f1": [],
                     "auc": []}
        kfold = KFold(n_splits, shuffle=True, random_state=self.random_state)

        for train_indices, test_indices in kfold.split(X):
            X_train = X.iloc[train_indices]
            X_val = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_val = y.iloc[test_indices]
            curr_model = HeartDiseaseClassifier(C=self.C, 
                                                random_state=self.random_state)
            curr_model.fit(X_train, y_train)
            metrics_dict = curr_model.evaluate(X_val, y_val)
            for metric in cv_results.keys():
                cv_results[metric].append(metrics_dict[metric])
        return cv_results