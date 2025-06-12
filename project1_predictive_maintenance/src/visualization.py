"""
Visualization Module for Predictive Maintenance System

This module provides visualization functions for:
- Sensor data analysis
- Model performance
- Prediction results
- System health monitoring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MaintenanceVisualizer:
    """
    A class for creating visualizations for the predictive maintenance system.
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        
    def plot_sensor_data_overview(self, data, save_path=None):
        """
        Create an overview plot of all sensor data.
        
        Args:
            data (pd.DataFrame): Sensor data
            save_path (str): Path to save the plot
        """
        sensor_columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
        available_sensors = [col for col in sensor_columns if col in data.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, sensor in enumerate(available_sensors):
            if i < 4:  # Only plot first 4 sensors
                axes[i].plot(data.index, data[sensor], alpha=0.7)
                axes[i].set_title(f'{sensor.replace("_", " ").title()} Over Time')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(sensor.replace("_", " ").title())
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_sensors), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_failure_distribution(self, data, save_path=None):
        """
        Plot the distribution of failures over time.
        
        Args:
            data (pd.DataFrame): Data with failure column
            save_path (str): Path to save the plot
        """
        if 'failure' not in data.columns:
            print("No failure column found in data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Failure count over time
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            daily_failures = data.groupby(data['timestamp'].dt.date)['failure'].sum()
            ax1.plot(daily_failures.index, daily_failures.values, marker='o')
            ax1.set_title('Daily Failure Count')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Number of Failures')
            ax1.tick_params(axis='x', rotation=45)
        else:
            # If no timestamp, just show failure distribution
            failure_counts = data['failure'].value_counts()
            ax1.bar(['No Failure', 'Failure'], failure_counts.values)
            ax1.set_title('Failure Distribution')
            ax1.set_ylabel('Count')
        
        # Failure rate by sensor ranges
        sensor_columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
        available_sensors = [col for col in sensor_columns if col in data.columns]
        
        if available_sensors:
            sensor = available_sensors[0]  # Use first available sensor
            data[f'{sensor}_binned'] = pd.cut(data[sensor], bins=10)
            failure_rate_by_bin = data.groupby(f'{sensor}_binned')['failure'].mean()
            
            ax2.bar(range(len(failure_rate_by_bin)), failure_rate_by_bin.values)
            ax2.set_title(f'Failure Rate by {sensor.title()} Range')
            ax2.set_xlabel(f'{sensor.title()} Bins')
            ax2.set_ylabel('Failure Rate')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sensor_correlations(self, data, save_path=None):
        """
        Plot correlation matrix of sensor data.
        
        Args:
            data (pd.DataFrame): Sensor data
            save_path (str): Path to save the plot
        """
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Sensor Data Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sensor_distributions(self, data, save_path=None):
        """
        Plot distributions of sensor readings by failure status.
        
        Args:
            data (pd.DataFrame): Data with sensor readings and failure column
            save_path (str): Path to save the plot
        """
        if 'failure' not in data.columns:
            print("No failure column found in data")
            return
        
        sensor_columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
        available_sensors = [col for col in sensor_columns if col in data.columns]
        
        n_sensors = len(available_sensors)
        if n_sensors == 0:
            print("No sensor columns found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, sensor in enumerate(available_sensors[:4]):
            # Plot distributions for failure and no-failure cases
            no_failure_data = data[data['failure'] == 0][sensor]
            failure_data = data[data['failure'] == 1][sensor]
            
            axes[i].hist(no_failure_data, alpha=0.7, label='No Failure', bins=30, density=True)
            axes[i].hist(failure_data, alpha=0.7, label='Failure', bins=30, density=True)
            axes[i].set_title(f'{sensor.replace("_", " ").title()} Distribution')
            axes[i].set_xlabel(sensor.replace("_", " ").title())
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_sensors, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_performance(self, y_true, y_pred, y_pred_proba, save_path=None):
        """
        Plot comprehensive model performance metrics.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_pred_proba (array): Predicted probabilities
            save_path (str): Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Prediction Probability Distribution
        no_failure_probs = y_pred_proba[y_true == 0]
        failure_probs = y_pred_proba[y_true == 1]
        ax3.hist(no_failure_probs, alpha=0.7, label='No Failure', bins=30, density=True)
        ax3.hist(failure_probs, alpha=0.7, label='Failure', bins=30, density=True)
        ax3.set_xlabel('Predicted Probability')
        ax3.set_ylabel('Density')
        ax3.set_title('Prediction Probability Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Threshold Analysis
        thresholds = np.linspace(0, 1, 100)
        precision_scores = []
        recall_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculate precision and recall
            tp = np.sum((y_pred_thresh == 1) & (y_true == 1))
            fp = np.sum((y_pred_thresh == 1) & (y_true == 0))
            fn = np.sum((y_pred_thresh == 0) & (y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        ax4.plot(thresholds, precision_scores, label='Precision', linewidth=2)
        ax4.plot(thresholds, recall_scores, label='Recall', linewidth=2)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Score')
        ax4.set_title('Precision-Recall vs Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_prediction_timeline(self, data, predictions, save_path=None):
        """
        Plot predictions over time with sensor data.
        
        Args:
            data (pd.DataFrame): Original data with timestamp
            predictions (array): Model predictions
            save_path (str): Path to save the plot
        """
        if 'timestamp' not in data.columns:
            print("No timestamp column found")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot sensor data
        sensor_columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
        available_sensors = [col for col in sensor_columns if col in data.columns]
        
        for sensor in available_sensors[:2]:  # Plot first 2 sensors
            ax1.plot(data['timestamp'], data[sensor], label=sensor.title(), alpha=0.7)
        
        ax1.set_ylabel('Sensor Values')
        ax1.set_title('Sensor Data Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot predictions
        ax2.scatter(data['timestamp'], predictions, c=predictions, cmap='RdYlBu_r', alpha=0.6)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Failure Probability')
        ax2.set_title('Failure Predictions Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_dashboard(self, data, model_results=None, save_path=None):
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            data (pd.DataFrame): Sensor data
            model_results (dict): Model evaluation results
            save_path (str): Path to save the dashboard
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Sensor data overview (top row)
        sensor_columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
        available_sensors = [col for col in sensor_columns if col in data.columns]
        
        for i, sensor in enumerate(available_sensors[:4]):
            ax = fig.add_subplot(gs[0, i])
            ax.plot(data.index[-1000:], data[sensor].iloc[-1000:], alpha=0.7)  # Last 1000 points
            ax.set_title(f'{sensor.title()}')
            ax.grid(True, alpha=0.3)
        
        # Failure distribution (middle left)
        if 'failure' in data.columns:
            ax = fig.add_subplot(gs[1, :2])
            failure_counts = data['failure'].value_counts()
            ax.pie(failure_counts.values, labels=['No Failure', 'Failure'], autopct='%1.1f%%')
            ax.set_title('Failure Distribution')
        
        # Correlation heatmap (middle right)
        ax = fig.add_subplot(gs[1, 2:])
        numeric_data = data[available_sensors].corr()
        sns.heatmap(numeric_data, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Sensor Correlations')
        
        # Model performance (bottom row)
        if model_results:
            # ROC Curve
            ax = fig.add_subplot(gs[2, :2])
            fpr, tpr, _ = roc_curve(model_results['y_true'], model_results['y_pred_proba'])
            auc_score = roc_auc_score(model_results['y_true'], model_results['y_pred_proba'])
            ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Model Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Confusion Matrix
            ax = fig.add_subplot(gs[2, 2:])
            cm = confusion_matrix(model_results['y_true'], model_results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.suptitle('Predictive Maintenance Dashboard', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """
    Main function to demonstrate visualization capabilities.
    """
    visualizer = MaintenanceVisualizer()
    
    # Try to load data
    try:
        data = pd.read_csv('data/processed/processed_sensor_data.csv')
        print(f"Loaded data with shape: {data.shape}")
        
        # Create visualizations
        print("Creating sensor data overview...")
        visualizer.plot_sensor_data_overview(data, 'models/sensor_overview.png')
        
        print("Creating failure distribution plot...")
        visualizer.plot_failure_distribution(data, 'models/failure_distribution.png')
        
        print("Creating sensor correlations plot...")
        visualizer.plot_sensor_correlations(data, 'models/sensor_correlations.png')
        
        print("Creating sensor distributions plot...")
        visualizer.plot_sensor_distributions(data, 'models/sensor_distributions.png')
        
        print("Visualizations created successfully!")
        
    except FileNotFoundError:
        print("Data file not found. Please run data_preprocessing.py first.")

if __name__ == "__main__":
    main()

