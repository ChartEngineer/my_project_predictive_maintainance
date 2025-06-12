"""
Feature Engineering Module for Predictive Maintenance System

This module creates advanced features from raw sensor data including:
- Rolling statistics (mean, std, min, max)
- Lag features
- Rate of change features
- Frequency domain features
- Statistical features
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    A class for creating advanced features from sensor data.
    """
    
    def __init__(self, window_sizes=[6, 12, 24]):
        self.window_sizes = window_sizes
        
    def create_rolling_features(self, data, columns=None):
        """
        Create rolling statistical features.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create rolling features for
            
        Returns:
            pd.DataFrame: Data with rolling features
        """
        if columns is None:
            columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
            
        feature_data = data.copy()
        
        for col in columns:
            if col in data.columns:
                for window in self.window_sizes:
                    # Rolling mean
                    feature_data[f'{col}_rolling_mean_{window}h'] = data[col].rolling(window=window).mean()
                    
                    # Rolling standard deviation
                    feature_data[f'{col}_rolling_std_{window}h'] = data[col].rolling(window=window).std()
                    
                    # Rolling min and max
                    feature_data[f'{col}_rolling_min_{window}h'] = data[col].rolling(window=window).min()
                    feature_data[f'{col}_rolling_max_{window}h'] = data[col].rolling(window=window).max()
                    
                    # Rolling range
                    feature_data[f'{col}_rolling_range_{window}h'] = (
                        feature_data[f'{col}_rolling_max_{window}h'] - 
                        feature_data[f'{col}_rolling_min_{window}h']
                    )
        
        print(f"Rolling features created for windows: {self.window_sizes}")
        return feature_data
    
    def create_lag_features(self, data, columns=None, lags=[1, 3, 6, 12]):
        """
        Create lag features.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create lag features for
            lags (list): Lag periods
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        if columns is None:
            columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
            
        feature_data = data.copy()
        
        for col in columns:
            if col in data.columns:
                for lag in lags:
                    feature_data[f'{col}_lag_{lag}h'] = data[col].shift(lag)
        
        print(f"Lag features created for lags: {lags}")
        return feature_data
    
    def create_rate_of_change_features(self, data, columns=None, periods=[1, 3, 6]):
        """
        Create rate of change features.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create rate of change features for
            periods (list): Periods for rate of change calculation
            
        Returns:
            pd.DataFrame: Data with rate of change features
        """
        if columns is None:
            columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
            
        feature_data = data.copy()
        
        for col in columns:
            if col in data.columns:
                for period in periods:
                    # Rate of change
                    feature_data[f'{col}_roc_{period}h'] = data[col].pct_change(periods=period)
                    
                    # Absolute rate of change
                    feature_data[f'{col}_abs_roc_{period}h'] = np.abs(feature_data[f'{col}_roc_{period}h'])
        
        print(f"Rate of change features created for periods: {periods}")
        return feature_data
    
    def create_statistical_features(self, data, columns=None):
        """
        Create statistical features for each sensor.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create statistical features for
            
        Returns:
            pd.DataFrame: Data with statistical features
        """
        if columns is None:
            columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
            
        feature_data = data.copy()
        
        for col in columns:
            if col in data.columns:
                # Skewness
                feature_data[f'{col}_skewness'] = data[col].rolling(window=24).apply(lambda x: stats.skew(x))
                
                # Kurtosis
                feature_data[f'{col}_kurtosis'] = data[col].rolling(window=24).apply(lambda x: stats.kurtosis(x))
                
                # Coefficient of variation
                rolling_mean = data[col].rolling(window=24).mean()
                rolling_std = data[col].rolling(window=24).std()
                feature_data[f'{col}_cv'] = rolling_std / rolling_mean
        
        print("Statistical features created")
        return feature_data
    
    def create_interaction_features(self, data):
        """
        Create interaction features between sensors.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with interaction features
        """
        feature_data = data.copy()
        
        # Temperature-Pressure interaction
        if 'temperature' in data.columns and 'pressure' in data.columns:
            feature_data['temp_pressure_ratio'] = data['temperature'] / data['pressure']
            feature_data['temp_pressure_product'] = data['temperature'] * data['pressure']
        
        # Vibration-Speed interaction
        if 'vibration' in data.columns and 'rotation_speed' in data.columns:
            feature_data['vibration_speed_ratio'] = data['vibration'] / data['rotation_speed']
            feature_data['vibration_speed_product'] = data['vibration'] * data['rotation_speed']
        
        # Overall health score (simple combination)
        if all(col in data.columns for col in ['temperature', 'vibration', 'pressure', 'rotation_speed']):
            # Normalize each sensor to 0-1 scale for health score
            temp_norm = (data['temperature'] - data['temperature'].min()) / (data['temperature'].max() - data['temperature'].min())
            vib_norm = (data['vibration'] - data['vibration'].min()) / (data['vibration'].max() - data['vibration'].min())
            press_norm = (data['pressure'] - data['pressure'].min()) / (data['pressure'].max() - data['pressure'].min())
            speed_norm = (data['rotation_speed'] - data['rotation_speed'].min()) / (data['rotation_speed'].max() - data['rotation_speed'].min())
            
            # Health score (lower is better)
            feature_data['health_score'] = (temp_norm + vib_norm + (1 - press_norm) + np.abs(speed_norm - 0.5)) / 4
        
        print("Interaction features created")
        return feature_data
    
    def create_frequency_features(self, data, columns=None, window_size=24):
        """
        Create frequency domain features using FFT.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create frequency features for
            window_size (int): Window size for FFT
            
        Returns:
            pd.DataFrame: Data with frequency features
        """
        if columns is None:
            columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
            
        feature_data = data.copy()
        
        for col in columns:
            if col in data.columns:
                # Dominant frequency
                def get_dominant_frequency(x):
                    if len(x) < window_size:
                        return np.nan
                    fft_vals = np.abs(fft(x))
                    freqs = np.fft.fftfreq(len(x))
                    dominant_freq_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
                    return freqs[dominant_freq_idx]
                
                feature_data[f'{col}_dominant_freq'] = data[col].rolling(window=window_size).apply(get_dominant_frequency)
                
                # Spectral energy
                def get_spectral_energy(x):
                    if len(x) < window_size:
                        return np.nan
                    fft_vals = np.abs(fft(x))
                    return np.sum(fft_vals**2)
                
                feature_data[f'{col}_spectral_energy'] = data[col].rolling(window=window_size).apply(get_spectral_energy)
        
        print("Frequency domain features created")
        return feature_data
    
    def feature_engineering_pipeline(self, data, save_path=None):
        """
        Complete feature engineering pipeline.
        
        Args:
            data (pd.DataFrame): Input preprocessed data
            save_path (str): Path to save engineered features
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        print("Starting feature engineering pipeline...")
        
        # Create rolling features
        data = self.create_rolling_features(data)
        
        # Create lag features
        data = self.create_lag_features(data)
        
        # Create rate of change features
        data = self.create_rate_of_change_features(data)
        
        # Create statistical features
        data = self.create_statistical_features(data)
        
        # Create interaction features
        data = self.create_interaction_features(data)
        
        # Create frequency features
        data = self.create_frequency_features(data)
        
        # Remove rows with NaN values (due to rolling windows and lags)
        initial_shape = data.shape
        data = data.dropna()
        final_shape = data.shape
        
        print(f"Removed {initial_shape[0] - final_shape[0]} rows with NaN values")
        
        # Save engineered features
        if save_path:
            data.to_csv(save_path, index=False)
            print(f"Engineered features saved to {save_path}")
        
        print("Feature engineering completed successfully!")
        print(f"Final feature count: {data.shape[1]}")
        
        return data

def main():
    """
    Main function to run feature engineering.
    """
    # Load preprocessed data
    try:
        data = pd.read_csv('data/processed/processed_sensor_data.csv')
        print(f"Loaded preprocessed data with shape: {data.shape}")
    except FileNotFoundError:
        print("Preprocessed data not found. Please run data_preprocessing.py first.")
        return
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Run feature engineering pipeline
    engineered_data = feature_engineer.feature_engineering_pipeline(
        data, 
        save_path='data/processed/engineered_features.csv'
    )
    
    print(f"Final engineered data shape: {engineered_data.shape}")
    print(f"Feature columns: {engineered_data.columns.tolist()}")

if __name__ == "__main__":
    main()

