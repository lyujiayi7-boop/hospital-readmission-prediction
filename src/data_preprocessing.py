"""
Data Preprocessing Pipeline for Hospital Readmission Prediction
Cleans and prepares patient data for machine learning models
Author: Jiayi Lyu
"""

import pandas as pd
import numpy as np
import logging
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses patient data including cleaning, encoding, and normalization
    Handles missing values and outliers from multiple data sources
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load patient data from CSV"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} features")
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in dataset
        Uses median for numerical, mode for categorical
        """
        logger.info("Handling missing values...")
        
        initial_missing = df.isnull().sum().sum()
        
        # Numerical columns - fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled {col} with median: {median_val:.2f}")
        
        # Categorical columns - fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"  Filled {col} with mode: {mode_val}")
        
        final_missing = df.isnull().sum().sum()
        logger.info(f"Reduced missing values from {initial_missing} to {final_missing}")
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate records"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=['encounter_id'], keep='first')
        removed = initial_count - len(df)
        logger.info(f"Removed {removed} duplicate records")
        return df
    
    def handle_outliers(self, df, columns=None):
        """
        Handle outliers using IQR method
        Caps extreme values rather than removing them
        """
        if columns is None:
            columns = ['time_in_hospital', 'num_lab_procedures', 
                      'num_procedures', 'num_medications']
        
        logger.info("Handling outliers using IQR method...")
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                # Cap outliers
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                if outliers > 0:
                    logger.info(f"  Capped {outliers} outliers in {col}")
        
        return df
    
    def encode_categorical(self, df):
        """
        Encode categorical variables
        Uses label encoding for binary, one-hot for multi-class
        """
        logger.info("Encoding categorical variables...")
        
        # Binary categorical - use label encoding
        binary_cols = ['gender']
        for col in binary_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                logger.info(f"  Label encoded {col}")
        
        # Multi-class categorical - use one-hot encoding
        multi_class_cols = ['race', 'admission_type']
        for col in multi_class_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                logger.info(f"  One-hot encoded {col}")
        
        return df
    
    def create_features(self, df):
        """
        Create additional features for better model performance
        """
        logger.info("Creating derived features...")
        
        # Total prior visits
        df['total_prior_visits'] = (df['number_emergency'] + 
                                    df['number_inpatient'] + 
                                    df['number_outpatient'])
        
        # High risk flag (multiple prior admissions)
        df['high_risk_flag'] = (df['total_prior_visits'] >= 3).astype(int)
        
        # Medication intensity
        df['medication_intensity'] = df['num_medications'] / (df['time_in_hospital'] + 1)
        
        # Procedure complexity
        df['procedure_complexity'] = df['num_procedures'] * df['number_diagnoses']
        
        # Age groups (clinical significance)
        df['is_elderly'] = (df['age'] >= 65).astype(int)
        df['is_very_elderly'] = (df['age'] >= 75).astype(int)
        
        logger.info(f"  Created 7 new features")
        
        return df
    
    def preprocess_pipeline(self, input_path, output_path=None):
        """
        Complete preprocessing pipeline
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_data(input_path)
        
        # Preprocessing steps
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        df = self.create_features(df)
        df = self.encode_categorical(df)
        
        # Drop non-feature columns
        drop_cols = ['encounter_id', 'patient_id', 'admission_date']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Features: {list(df.columns)}")
        
        # Save processed data
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Saved processed data to {output_path}")
            
            # Save encoders and scaler
            joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
            logger.info("✓ Saved label encoders")
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess patient readmission data')
    parser.add_argument('--input', type=str, default='data/raw/patient_data.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/processed/cleaned_data.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_pipeline(args.input, args.output)