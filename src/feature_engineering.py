"""
Feature Engineering for Hospital Readmission Prediction
Creates advanced features to improve model performance
Author: Jiayi Lyu
"""

import pandas as pd
import numpy as np
import logging
import argparse
from sklearn.feature_selection import mutual_info_classif
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates and selects features for readmission prediction
    Based on clinical domain knowledge and statistical analysis
    """
    
    def __init__(self):
        self.feature_importance = {}
    
    def load_data(self, filepath):
        """Load preprocessed data"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} records")
        return df
    
    def create_age_features(self, df):
        """
        Create age-related features
        Age is a strong predictor of readmission
        """
        logger.info("Creating age-related features...")
        
        # Age squared (non-linear relationship)
        df['age_squared'] = df['age'] ** 2
        
        # Age groups based on clinical significance
        df['age_group_young'] = (df['age'] < 50).astype(int)
        df['age_group_middle'] = ((df['age'] >= 50) & (df['age'] < 65)).astype(int)
        df['age_group_senior'] = ((df['age'] >= 65) & (df['age'] < 75)).astype(int)
        df['age_group_elderly'] = (df['age'] >= 75).astype(int)
        
        logger.info("  Created 5 age features")
        return df
    
    def create_los_features(self, df):
        """
        Create length of stay (LOS) features
        LOS is highly correlated with readmission risk
        """
        logger.info("Creating length of stay features...")
        
        # Extended stay flags
        df['short_stay'] = (df['time_in_hospital'] <= 3).astype(int)
        df['medium_stay'] = ((df['time_in_hospital'] > 3) & 
                            (df['time_in_hospital'] <= 7)).astype(int)
        df['long_stay'] = (df['time_in_hospital'] > 7).astype(int)
        df['very_long_stay'] = (df['time_in_hospital'] > 14).astype(int)
        
        # LOS squared
        df['los_squared'] = df['time_in_hospital'] ** 2
        
        logger.info("  Created 5 LOS features")
        return df
    
    def create_utilization_features(self, df):
        """
        Create healthcare utilization features
        Prior utilization is a strong readmission predictor
        """
        logger.info("Creating healthcare utilization features...")
        
        # Total prior utilization
        df['total_prior_admissions'] = (df['number_emergency'] + 
                                        df['number_inpatient'])
        
        # Frequent flyer flag (multiple ED visits)
        df['frequent_ed_user'] = (df['number_emergency'] >= 2).astype(int)
        
        # Any prior admission
        df['has_prior_admission'] = (df['total_prior_admissions'] > 0).astype(int)
        
        # High utilization flag
        df['high_utilization'] = (df['total_prior_visits'] >= 5).astype(int)
        
        # Emergency ratio
        df['emergency_ratio'] = df['number_emergency'] / (df['total_prior_visits'] + 1)
        
        logger.info("  Created 5 utilization features")
        return df
    
    def create_clinical_complexity_features(self, df):
        """
        Create features representing clinical complexity
        More complex cases have higher readmission risk
        """
        logger.info("Creating clinical complexity features...")
        
        # Diagnosis burden
        df['high_diagnosis_count'] = (df['number_diagnoses'] >= 7).astype(int)
        df['diagnosis_squared'] = df['number_diagnoses'] ** 2
        
        # Medication burden
        df['high_medication_count'] = (df['num_medications'] >= 15).astype(int)
        df['polypharmacy'] = (df['num_medications'] >= 20).astype(int)
        
        # Procedure intensity
        df['has_procedures'] = (df['num_procedures'] > 0).astype(int)
        df['high_procedure_count'] = (df['num_procedures'] >= 3).astype(int)
        
        # Lab intensity
        df['high_lab_count'] = (df['num_lab_procedures'] >= 50).astype(int)
        
        logger.info("  Created 7 complexity features")
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between important variables
        Captures non-linear relationships
        """
        logger.info("Creating interaction features...")
        
        # Age × Prior admissions (elderly with prior admissions = very high risk)
        if 'total_prior_admissions' in df.columns:
            df['age_prior_interaction'] = df['age'] * df['total_prior_admissions']
        
        # Age × Medications (elderly on many meds = high risk)
        df['age_medication_interaction'] = df['age'] * df['num_medications']
        
        # LOS × Diagnoses (long stay + many diagnoses = complex case)
        df['los_diagnosis_interaction'] = (df['time_in_hospital'] * 
                                           df['number_diagnoses'])
        
        # LOS × Procedures
        df['los_procedure_interaction'] = (df['time_in_hospital'] * 
                                           df['num_procedures'])
        
        # Medications × Diagnoses (medication complexity)
        df['med_diagnosis_interaction'] = (df['num_medications'] * 
                                           df['number_diagnoses'])
        
        logger.info("  Created 5 interaction features")
        return df
    
    def create_risk_scores(self, df):
        """
        Create composite risk scores based on clinical knowledge
        """
        logger.info("Creating composite risk scores...")
        
        # Readmission risk score (weighted sum of key predictors)
        risk_score = 0
        
        # Age component (20% weight)
        if 'age' in df.columns:
            age_normalized = df['age'] / 100
            risk_score += 0.20 * age_normalized
        
        # Prior admission component (30% weight)
        if 'total_prior_admissions' in df.columns:
            prior_normalized = df['total_prior_admissions'] / df['total_prior_admissions'].max()
            risk_score += 0.30 * prior_normalized
        
        # LOS component (20% weight)
        los_normalized = df['time_in_hospital'] / df['time_in_hospital'].max()
        risk_score += 0.20 * los_normalized
        
        # Diagnosis burden component (15% weight)
        diag_normalized = df['number_diagnoses'] / df['number_diagnoses'].max()
        risk_score += 0.15 * diag_normalized
        
        # Medication burden component (15% weight)
        med_normalized = df['num_medications'] / df['num_medications'].max()
        risk_score += 0.15 * med_normalized
        
        df['composite_risk_score'] = risk_score
        
        # High risk flag
        df['very_high_risk'] = (risk_score > 0.6).astype(int)
        
        logger.info("  Created composite risk score")
        return df
    
    def select_top_features(self, df, target_col='readmitted', n_features=30):
        """
        Select top features using mutual information
        """
        logger.info(f"Selecting top {n_features} features...")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Store feature importance
        self.feature_importance = dict(zip(feature_importance_df['feature'], 
                                           feature_importance_df['importance']))
        
        # Select top features
        top_features = feature_importance_df.head(n_features)['feature'].tolist()
        
        logger.info("\nTop 10 Most Important Features:")
        for i, row in feature_importance_df.head(10).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Keep target column
        selected_columns = top_features + [target_col]
        df_selected = df[selected_columns]
        
        logger.info(f"\nReduced features from {len(X.columns)} to {len(top_features)}")
        
        return df_selected, feature_importance_df
    
    def engineer_features(self, input_path, output_path=None, select_features=True):
        """
        Complete feature engineering pipeline
        """
        logger.info("=" * 60)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_data(input_path)
        initial_features = len(df.columns) - 1  # Exclude target
        
        # Create features
        df = self.create_age_features(df)
        df = self.create_los_features(df)
        df = self.create_utilization_features(df)
        df = self.create_clinical_complexity_features(df)
        df = self.create_interaction_features(df)
        df = self.create_risk_scores(df)
        
        total_features = len(df.columns) - 1  # Exclude target
        new_features = total_features - initial_features
        
        logger.info(f"\nFeature Summary:")
        logger.info(f"  Initial features: {initial_features}")
        logger.info(f"  New features created: {new_features}")
        logger.info(f"  Total features: {total_features}")
        
        # Feature selection
        if select_features and 'readmitted' in df.columns:
            df, importance_df = self.select_top_features(df)
            
            # Save feature importance
            if output_path:
                importance_path = output_path.replace('.csv', '_importance.csv')
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"✓ Saved feature importance to {importance_path}")
        
        # Save engineered features
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Saved engineered features to {output_path}")
        
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 60)
        
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Engineer features for readmission prediction')
    parser.add_argument('--input', type=str, default='data/processed/cleaned_data.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/processed/features.csv',
                       help='Output CSV file path')
    parser.add_argument('--no-selection', action='store_true',
                       help='Skip feature selection')
    
    args = parser.parse_args()
    
    engineer = FeatureEngineer()
    engineer.engineer_features(args.input, args.output, 
                              select_features=not args.no_selection)