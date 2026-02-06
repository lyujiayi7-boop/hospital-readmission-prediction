"""
Real-time Prediction Service for Hospital Readmission
Loads trained model and makes predictions on new patient data
Author: Jiayi Lyu
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Union
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReadmissionPredictor:
    """
    Makes real-time predictions for patient readmission risk
    Used in production dashboard for clinical decision support
    """
    
    def __init__(self, model_path='models/best_model.pkl'):
        """
        Load trained model
        
        Args:
            model_path: Path to the trained model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        logger.info("✓ Model loaded successfully")
        
        # Load feature names if available
        self.expected_features = None
    
    def predict_single_patient(self, patient_data: Dict) -> Dict:
        """
        Predict readmission risk for a single patient
        
        Args:
            patient_data: Dictionary with patient features
            
        Returns:
            Dictionary with prediction and probability
        """
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0, 1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        result = {
            'readmission_risk': int(prediction),
            'readmission_probability': float(probability),
            'risk_level': risk_level,
            'risk_percentage': f"{probability * 100:.1f}%"
        }
        
        logger.info(f"Prediction: {risk_level} risk ({probability*100:.1f}%)")
        
        return result
    
    def predict_batch(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict readmission risk for multiple patients
        
        Args:
            patient_df: DataFrame with patient features
            
        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Making predictions for {len(patient_df)} patients...")
        
        # Make predictions
        predictions = self.model.predict(patient_df)
        probabilities = self.model.predict_proba(patient_df)[:, 1]
        
        # Add to dataframe
        patient_df['readmission_prediction'] = predictions
        patient_df['readmission_probability'] = probabilities
        
        # Risk levels
        patient_df['risk_level'] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Moderate', 'High']
        )
        
        logger.info("✓ Predictions complete")
        
        return patient_df
    
    def get_high_risk_patients(self, patient_df: pd.DataFrame, 
                               threshold: float = 0.6) -> pd.DataFrame:
        """
        Identify high-risk patients requiring intervention
        
        Args:
            patient_df: DataFrame with predictions
            threshold: Probability threshold for high risk
            
        Returns:
            DataFrame with only high-risk patients
        """
        high_risk = patient_df[patient_df['readmission_probability'] >= threshold]
        
        logger.info(f"Identified {len(high_risk)} high-risk patients "
                   f"({len(high_risk)/len(patient_df)*100:.1f}%)")
        
        return high_risk.sort_values('readmission_probability', ascending=False)


# Example usage
if __name__ == "__main__":
    # Example patient data
    sample_patient = {
        'age': 72,
        'gender': 1,
        'time_in_hospital': 8,
        'num_lab_procedures': 45,
        'num_procedures': 3,
        'num_medications': 18,
        'number_outpatient': 2,
        'number_emergency': 1,
        'number_inpatient': 1,
        'number_diagnoses': 7
    }
    
    try:
        predictor = ReadmissionPredictor()
        result = predictor.predict_single_patient(sample_patient)
        
        print("\n" + "="*50)
        print("PATIENT READMISSION RISK ASSESSMENT")
        print("="*50)
        print(f"Risk Level: {result['risk_level']}")
        print(f"Probability: {result['risk_percentage']}")
        print(f"Recommendation: {'Intervention recommended' if result['readmission_risk'] == 1 else 'Standard care'}")
        print("="*50)
        
    except FileNotFoundError:
        print("Model not found. Please train the model first:")
        print("  python src/model_training.py")