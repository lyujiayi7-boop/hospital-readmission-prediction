"""
Generate Synthetic Patient Data for Demonstration
Creates realistic hospital readmission dataset
Author: Jiayi Lyu
Date: 2024-2025
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_patient_records(n_patients: int = 50000) -> pd.DataFrame:
    """
    Generate synthetic patient demographic data
    
    Args:
        n_patients: Number of unique patients to generate
        
    Returns:
        DataFrame with patient demographics
    """
    logger.info(f"Generating {n_patients} patient records...")
    
    np.random.seed(42)
    
    data = {
        'patient_id': [f'PT{str(i).zfill(7)}' for i in range(1, n_patients + 1)],
        'age': np.random.choice(range(18, 95), size=n_patients, 
                                p=np.array([0.15] + [0.85/76]*76)),  # Skewed towards older
        'gender': np.random.choice(['Male', 'Female'], size=n_patients, p=[0.47, 0.53]),
        'race': np.random.choice(
            ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'],
            size=n_patients,
            p=[0.58, 0.22, 0.13, 0.05, 0.02]
        ),
        'weight': np.random.normal(180, 40, n_patients).clip(100, 350).astype(int),
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Created {len(df)} patient demographic records")
    return df


def generate_admissions(patients_df: pd.DataFrame, avg_admissions_per_patient: float = 2.1) -> pd.DataFrame:
    """
    Generate hospital admission records
    Each patient can have multiple admissions
    
    Args:
        patients_df: DataFrame with patient information
        avg_admissions_per_patient: Average number of admissions per patient
        
    Returns:
        DataFrame with admission records
    """
    logger.info("Generating hospital admission records...")
    
    np.random.seed(42)
    
    admissions_list = []
    encounter_id = 1
    
    for idx, patient in patients_df.iterrows():
        # Number of admissions for this patient (Poisson distribution)
        n_admissions = np.random.poisson(avg_admissions_per_patient)
        n_admissions = max(1, min(n_admissions, 8))  # At least 1, max 8
        
        for admission_num in range(n_admissions):
            # Generate admission date (within last 2 years)
            days_ago = np.random.randint(1, 730)
            admission_date = datetime.now() - timedelta(days=days_ago)
            
            # Length of stay
            los = max(1, int(np.random.gamma(2.5, 1.5)))  # Gamma distribution for LOS
            
            # Previous admission counts
            prev_emergency = np.random.poisson(0.3) if admission_num > 0 else 0
            prev_inpatient = admission_num
            prev_outpatient = np.random.poisson(1.2)
            
            # Clinical metrics
            num_lab_procedures = int(np.random.gamma(5, 5)) + 10
            num_procedures = np.random.poisson(1.8)
            num_medications = int(np.random.gamma(2, 5)) + 5
            num_diagnoses = min(16, max(1, int(np.random.gamma(1.8, 2))))
            
            # Admission type
            if admission_num == 0:
                admission_type = np.random.choice(
                    ['Emergency', 'Urgent', 'Elective', 'Trauma'],
                    p=[0.52, 0.28, 0.17, 0.03]
                )
            else:
                # If readmission, more likely to be emergency
                admission_type = np.random.choice(
                    ['Emergency', 'Urgent', 'Elective', 'Trauma'],
                    p=[0.68, 0.25, 0.05, 0.02]
                )
            
            # Calculate readmission probability based on risk factors
            readmit_prob = 0.11  # Base rate ~11%
            
            # Risk factors (realistic clinical predictors)
            if patient['age'] > 65:
                readmit_prob += 0.08
            if patient['age'] > 75:
                readmit_prob += 0.06
            if los > 7:
                readmit_prob += 0.09
            if los > 14:
                readmit_prob += 0.07
            if prev_inpatient > 0:
                readmit_prob += 0.12
            if prev_emergency > 0:
                readmit_prob += 0.08
            if num_diagnoses > 6:
                readmit_prob += 0.06
            if num_medications > 15:
                readmit_prob += 0.05
            if admission_type == 'Emergency':
                readmit_prob += 0.07
            
            readmit_prob = min(0.65, readmit_prob)  # Cap at 65%
            
            # Determine if readmitted (only if not the last admission)
            if admission_num < n_admissions - 1:
                readmitted = 1 if np.random.random() < readmit_prob else 0
            else:
                # For last admission, use the calculated probability
                readmitted = 1 if np.random.random() < readmit_prob else 0
            
            admission = {
                'encounter_id': f'ENC{str(encounter_id).zfill(8)}',
                'patient_id': patient['patient_id'],
                'admission_date': admission_date.strftime('%Y-%m-%d'),
                'admission_type': admission_type,
                'time_in_hospital': los,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': prev_outpatient,
                'number_emergency': prev_emergency,
                'number_inpatient': prev_inpatient,
                'number_diagnoses': num_diagnoses,
                'readmitted': readmitted
            }
            
            admissions_list.append(admission)
            encounter_id += 1
    
    df = pd.DataFrame(admissions_list)
    readmission_rate = df['readmitted'].mean()
    logger.info(f"Created {len(df)} admission records")
    logger.info(f"Overall readmission rate: {readmission_rate:.2%}")
    
    return df


def create_database_and_csv(n_patients: int = 50000):
    """
    Create complete dataset and save to both SQLite and CSV
    
    Args:
        n_patients: Number of patients to generate
    """
    logger.info("=" * 60)
    logger.info("HOSPITAL READMISSION DATA GENERATION")
    logger.info("=" * 60)
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate data
    patients_df = generate_patient_records(n_patients)
    admissions_df = generate_admissions(patients_df)
    
    # Merge for complete dataset
    complete_df = admissions_df.merge(patients_df, on='patient_id', how='left')
    
    # Reorder columns for better readability
    column_order = [
        'encounter_id', 'patient_id', 'age', 'gender', 'race', 'weight',
        'admission_date', 'admission_type', 'time_in_hospital',
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'readmitted'
    ]
    complete_df = complete_df[column_order]
    
    # Save to CSV
    csv_path = 'data/raw/patient_data.csv'
    complete_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved complete dataset to {csv_path}")
    
    # Also save to SQLite database for SQL demonstration
    db_path = 'data/hospital_data.db'
    conn = sqlite3.connect(db_path)
    
    # Save patients table
    patients_df.to_sql('patients', conn, if_exists='replace', index=False)
    
    # Save admissions table
    admissions_df.to_sql('admissions', conn, if_exists='replace', index=False)
    
    conn.close()
    logger.info(f"✓ Saved database to {db_path}")
    
    # Print summary statistics
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total patients: {len(patients_df):,}")
    logger.info(f"Total admissions: {len(admissions_df):,}")
    logger.info(f"Average admissions per patient: {len(admissions_df)/len(patients_df):.2f}")
    logger.info(f"Readmission rate: {complete_df['readmitted'].mean():.2%}")
    logger.info(f"Average age: {complete_df['age'].mean():.1f} years")
    logger.info(f"Average length of stay: {complete_df['time_in_hospital'].mean():.1f} days")
    logger.info(f"Average medications: {complete_df['num_medications'].mean():.1f}")
    logger.info("=" * 60)
    
    # Display age distribution
    logger.info("\nAge Distribution:")
    age_bins = [0, 30, 50, 65, 75, 100]
    age_labels = ['18-29', '30-49', '50-64', '65-74', '75+']
    complete_df['age_group'] = pd.cut(complete_df['age'], bins=age_bins, labels=age_labels)
    print(complete_df['age_group'].value_counts().sort_index())
    
    # Display readmission by age group
    logger.info("\nReadmission Rate by Age Group:")
    readmit_by_age = complete_df.groupby('age_group')['readmitted'].agg(['mean', 'count'])
    readmit_by_age['mean'] = readmit_by_age['mean'].apply(lambda x: f"{x:.2%}")
    print(readmit_by_age)
    
    return complete_df


if __name__ == "__main__":
    # Generate dataset with 50,000 patients (will create ~100,000+ records)
    # Adjust n_patients to create larger datasets (e.g., 500,000 for 1M+ records)
    df = create_database_and_csv(n_patients=500000)  # This will create ~1M records
    
    logger.info("\n✓ Data generation complete!")
    logger.info("Next steps:")
    logger.info("1. Run data preprocessing: python src/data_preprocessing.py")
    logger.info("2. Run feature engineering: python src/feature_engineering.py")
    logger.info("3. Train models: python src/model_training.py")