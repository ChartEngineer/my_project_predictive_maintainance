"""
Model Training Module for Predictive Maintenance System

This module handles:
- Model selection and training
- Hyperparameter tuning
- Model evaluation
- Model persistence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    A class for training and evaluating machine learning models for predictive maintenance.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """
        Load engineered features data.
        
        Args:
            file_path (str): Path to the engineered features CSV file
            
        Returns:
            tuple: X (features), y (target)
        """
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {data.shape}")
            
            # Separate features and target
            X = data.drop(['failure', 'timestamp'], axis=1, errors='ignore')
            y = data['failure']
            
            print(f"Features shape: {X.shape}")
            print(f"Target distribution: {y.value_counts()}")
            
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test set size
            val_size (float): Validation set size
            
        Returns:
            tuple: Train, validation, and test sets
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Train set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def initialize_models(self):
        """
        Initialize different machine learning models.
        """
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            ),
            'svm': SVC(
                random_state=self.random_state,
                class_weight='balanced',
                probability=True
            )
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train all models and evaluate on validation set.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            
        Returns:
            dict: Model performance results
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name} - AUC: {auc_score:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """
        Perform hyperparameter tuning for the best performing model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_name (str): Name of the model to tune
            
        Returns:
            object: Best model after tuning
        """
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
            
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            base_model = GradientBoostingClassifier(random_state=self.random_state)
        
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return self.models[model_name]
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model on test set and generate detailed metrics.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{model_name} Test Set Evaluation:")
        print("="*50)
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'auc_score': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def plot_roc_curve(self, y_test, y_pred_proba, model_name="Model"):
        """
        Plot ROC curve.
        
        Args:
            y_test (pd.Series): True labels
            y_pred_proba (np.array): Predicted probabilities
            model_name (str): Name of the model
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n=20):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model
            feature_names (list): List of feature names
            top_n (int): Number of top features to display
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Model does not have feature_importances_ attribute")
    
    def save_model(self, model, file_path):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            file_path (str): Path to save the model
        """
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
    
    def training_pipeline(self, data_path='data/processed/engineered_features.csv'):
        """
        Complete model training pipeline.
        
        Args:
            data_path (str): Path to the engineered features data
            
        Returns:
            object: Best trained model
        """
        print("Starting model training pipeline...")
        
        # Load data
        X, y = self.load_data(data_path)
        if X is None:
            return None
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y)
        
        # Initialize models
        self.initialize_models()
        
        # Train models
        results = self.train_models(X_train, y_train, X_val, y_val)
        
        # Select best model based on validation AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
        print(f"\nBest model: {best_model_name}")
        
        # Hyperparameter tuning for best model
        tuned_model = self.hyperparameter_tuning(X_train, y_train, best_model_name)
        
        # Final evaluation on test set
        test_results = self.evaluate_model(tuned_model, X_test, y_test, best_model_name)
        
        # Plot ROC curve
        self.plot_roc_curve(y_test, test_results['y_pred_proba'], best_model_name)
        
        # Plot feature importance
        self.plot_feature_importance(tuned_model, X.columns)
        
        # Save best model
        self.save_model(tuned_model, 'models/best_model.pkl')
        
        # Save scaler (if used)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        self.best_model = tuned_model
        print("Model training pipeline completed successfully!")
        
        return tuned_model

def main():
    """
    Main function to run model training.
    """
    trainer = ModelTrainer()
    best_model = trainer.training_pipeline()
    
    if best_model:
        print("Model training completed successfully!")
    else:
        print("Model training failed!")

if __name__ == "__main__":
    main()

