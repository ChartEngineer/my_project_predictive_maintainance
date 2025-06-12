"""
Data Preprocessing Module for Predictive Maintenance System

This module handles the preprocessing of industrial sensor data including:
- Data cleaning and validation
- Handling missing values
- Outlier detection and treatment
- Data normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A class for preprocessing industrial sensor data for predictive maintenance.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.outlier_threshold = 3  # Z-score threshold for outlier detection
        
    def load_data(self, file_path):
        """
        Load sensor data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def generate_sample_data(self, n_samples=10000):
        """
        Generate sample industrial sensor data for demonstration.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated sensor data
        """
        np.random.seed(42)
        
        # Generate time series data
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        
        # Generate sensor readings with realistic patterns
        temperature = 70 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 2, n_samples)
        vibration = 0.5 + 0.2 * np.sin(np.arange(n_samples) * 2 * np.pi / 168) + np.random.normal(0, 0.1, n_samples)
        pressure = 100 + 5 * np.cos(np.arange(n_samples) * 2 * np.pi / 12) + np.random.normal(0, 1, n_samples)
        rotation_speed = 1800 + 50 * np.sin(np.arange(n_samples) * 2 * np.pi / 6) + np.random.normal(0, 10, n_samples)
        
        # Generate failure labels (1% failure rate)
        failure_probability = 0.01 + 0.005 * (temperature > 85) + 0.01 * (vibration > 0.8) + 0.005 * (pressure < 95)
        failure = np.random.binomial(1, failure_probability)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'vibration': vibration,
            'pressure': pressure,
            'rotation_speed': rotation_speed,
            'failure': failure
        })
        
        return data
    
    def handle_missing_values(self, data, strategy='interpolate'):
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            strategy (str): Strategy for handling missing values
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        if strategy == 'interpolate':
            # Use linear interpolation for time series data
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].interpolate(method='linear')
        elif strategy == 'forward_fill':
            data = data.fillna(method='ffill')
        elif strategy == 'drop':
            data = data.dropna()
            
        print(f"Missing values handled using {strategy} strategy")
        return data
    
    def detect_outliers(self, data, columns=None):
        """
        Detect outliers using Z-score method.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to check for outliers
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        outlier_mask = pd.DataFrame(False, index=data.index, columns=columns)
        
        for col in columns:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_mask[col] = z_scores > self.outlier_threshold
                
        return outlier_mask
    
    def treat_outliers(self, data, outlier_mask, method='clip'):
        """
        Treat outliers in the data.
        
        Args:
            data (pd.DataFrame): Input data
            outlier_mask (pd.DataFrame): Boolean mask indicating outliers
            method (str): Method for treating outliers
            
        Returns:
            pd.DataFrame: Data with outliers treated
        """
        data_treated = data.copy()
        
        for col in outlier_mask.columns:
            if method == 'clip':
                # Clip outliers to 99th percentile
                upper_bound = data[col].quantile(0.99)
                lower_bound = data[col].quantile(0.01)
                data_treated[col] = data_treated[col].clip(lower_bound, upper_bound)
            elif method == 'remove':
                # Remove rows with outliers
                data_treated = data_treated[~outlier_mask[col]]
                
        print(f"Outliers treated using {method} method")
        return data_treated
    
    def normalize_data(self, data, columns=None, fit=True):
        """
        Normalize numerical data using StandardScaler.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to normalize
            fit (bool): Whether to fit the scaler
            
        Returns:
            pd.DataFrame: Normalized data
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col != 'failure']  # Exclude target variable
            
        data_normalized = data.copy()
        
        if fit:
            data_normalized[columns] = self.scaler.fit_transform(data[columns])
        else:
            data_normalized[columns] = self.scaler.transform(data[columns])
            
        print(f"Data normalized for columns: {columns}")
        return data_normalized
    
    def preprocess_pipeline(self, data=None, save_path=None):
        """
        Complete preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Input data (if None, generates sample data)
            save_path (str): Path to save processed data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        print("Starting data preprocessing pipeline...")
        
        # Generate or load data
        if data is None:
            data = self.generate_sample_data()
            print("Sample data generated")
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Detect and treat outliers
        outlier_mask = self.detect_outliers(data)
        data = self.treat_outliers(data, outlier_mask)
        
        # Normalize data
        data = self.normalize_data(data)
        
        # Save processed data
        if save_path:
            data.to_csv(save_path, index=False)
            print(f"Processed data saved to {save_path}")
        
        print("Data preprocessing completed successfully!")
        return data

def main():
    """
    Main function to run data preprocessing.
    """
    preprocessor = DataPreprocessor()
    
    # Generate and preprocess sample data
    processed_data = preprocessor.preprocess_pipeline(
        save_path='data/processed/processed_sensor_data.csv'
    )
    
    print(f"Final data shape: {processed_data.shape}")
    print(f"Data columns: {processed_data.columns.tolist()}")
    print(f"Failure rate: {processed_data['failure'].mean():.3f}")

if __name__ == "__main__":
    main()

