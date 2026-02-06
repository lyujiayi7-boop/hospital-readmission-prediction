"""
SQL Data Extraction and Aggregation Module
Extracts patient data from multiple database sources and performs aggregations
Author: Jiayi Lyu
"""

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import logging
from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientDataExtractor:
    """
    Handles data extraction from hospital databases using SQL
    Processes over 1 million patient records from multiple sources
    """
    
    def __init__(self, db_config: Dict[str, str] = None):
        """
        Initialize database connection
        
        Args:
            db_config: Database configuration dictionary
        """
        if db_config is None:
            # Use environment variables or default to SQLite for demo
            self.db_type = os.getenv('DB_TYPE', 'sqlite')
            self.db_path = os.getenv('DB_PATH', 'data/hospital_data.db')
        else:
            self.db_type = db_config.get('type', 'sqlite')
            self.db_path = db_config.get('path', 'data/hospital_data.db')
        
        self.engine = self._create_connection()
        logger.info(f"Connected to {self.db_type} database")
    
    def _create_connection(self):
        """Create database connection"""
        if self.db_type == 'sqlite':
            connection_string = f'sqlite:///{self.db_path}'
        elif self.db_type == 'postgresql':
            user = os.getenv('DB_USER')
            password = os.getenv('DB_PASSWORD')
            host = os.getenv('DB_HOST', 'localhost')
            port = os.getenv('DB_PORT', '5432')
            database = os.getenv('DB_NAME')
            connection_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        elif self.db_type == 'mysql':
            user = os.getenv('DB_USER')
            password = os.getenv('DB_PASSWORD')
            host = os.getenv('DB_HOST', 'localhost')
            port = os.getenv('DB_PORT', '3306')
            database = os.getenv('DB_NAME')
            connection_string = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        return create_engine(connection_string)
    
    def extract_patient_demographics(self) -> pd.DataFrame:
        """
        Extract patient demographic information
        Uses SQL joins to combine data from multiple tables
        """
        query = """
        SELECT 
            p.patient_id,
            p.age,
            p.gender,
            p.race,
            p.weight,
            p.payer_code,
            a.admission_type_id,
            a.discharge_disposition_id,
            a.admission_source_id
        FROM patients p
        LEFT JOIN admissions a ON p.patient_id = a.patient_id
        WHERE a.admission_date >= DATE('now', '-2 years')
        ORDER BY p.patient_id
        """
        
        logger.info("Extracting patient demographics...")
        df = pd.read_sql_query(query, self.engine)
        logger.info(f"Extracted {len(df)} patient demographic records")
        return df
    
    def extract_admission_details(self) -> pd.DataFrame:
        """
        Extract detailed admission information with aggregations
        Demonstrates complex SQL with window functions and aggregations
        """
        query = """
        SELECT 
            a.patient_id,
            a.encounter_id,
            a.admission_type_id,
            a.time_in_hospital,
            a.num_lab_procedures,
            a.num_procedures,
            a.num_medications,
            a.number_outpatient,
            a.number_emergency,
            a.number_inpatient,
            a.number_diagnoses,
            
            -- Calculate rolling averages for previous admissions
            AVG(a.time_in_hospital) OVER (
                PARTITION BY a.patient_id 
                ORDER BY a.admission_date 
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as avg_prev_los,
            
            -- Count previous admissions
            COUNT(*) OVER (
                PARTITION BY a.patient_id 
                ORDER BY a.admission_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) as previous_admission_count,
            
            -- Flag for readmission
            CASE 
                WHEN readmitted IN ('YES', '<30') THEN 1 
                ELSE 0 
            END as readmitted
            
        FROM admissions a
        WHERE a.admission_date >= DATE('now', '-2 years')
        ORDER BY a.patient_id, a.admission_date
        """
        
        logger.info("Extracting admission details with aggregations...")
        df = pd.read_sql_query(query, self.engine)
        logger.info(f"Extracted {len(df)} admission records")
        return df
    
    def extract_diagnosis_data(self) -> pd.DataFrame:
        """
        Extract and aggregate diagnosis information
        Groups diagnosis codes by patient
        """
        query = """
        SELECT 
            d.patient_id,
            d.encounter_id,
            d.diag_1,
            d.diag_2,
            d.diag_3,
            
            -- Categorize primary diagnosis
            CASE 
                WHEN CAST(SUBSTR(d.diag_1, 1, 3) AS INTEGER) BETWEEN 390 AND 459 
                    OR d.diag_1 = '785' THEN 'Circulatory'
                WHEN CAST(SUBSTR(d.diag_1, 1, 3) AS INTEGER) BETWEEN 460 AND 519 
                    OR d.diag_1 = '786' THEN 'Respiratory'
                WHEN CAST(SUBSTR(d.diag_1, 1, 3) AS INTEGER) BETWEEN 520 AND 579 
                    OR d.diag_1 = '787' THEN 'Digestive'
                WHEN d.diag_1 LIKE '250%' THEN 'Diabetes'
                WHEN CAST(SUBSTR(d.diag_1, 1, 3) AS INTEGER) BETWEEN 800 AND 999 THEN 'Injury'
                WHEN CAST(SUBSTR(d.diag_1, 1, 3) AS INTEGER) BETWEEN 710 AND 739 THEN 'Musculoskeletal'
                WHEN CAST(SUBSTR(d.diag_1, 1, 3) AS INTEGER) BETWEEN 580 AND 629 
                    OR d.diag_1 = '788' THEN 'Genitourinary'
                WHEN CAST(SUBSTR(d.diag_1, 1, 3) AS INTEGER) BETWEEN 140 AND 239 THEN 'Neoplasms'
                ELSE 'Other'
            END as primary_diagnosis_category,
            
            -- Count unique diagnoses
            (CASE WHEN d.diag_1 IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN d.diag_2 IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN d.diag_3 IS NOT NULL THEN 1 ELSE 0 END) as diagnosis_count
             
        FROM diagnoses d
        """
        
        logger.info("Extracting diagnosis data...")
        df = pd.read_sql_query(query, self.engine)
        logger.info(f"Extracted {len(df)} diagnosis records")
        return df
    
    def extract_medication_data(self) -> pd.DataFrame:
        """
        Extract medication information
        Aggregates medication counts and changes
        """
        query = """
        SELECT 
            m.patient_id,
            m.encounter_id,
            
            -- Count medication changes
            SUM(CASE WHEN m.metformin IN ('Up', 'Down') THEN 1 ELSE 0 END +
                CASE WHEN m.repaglinide IN ('Up', 'Down') THEN 1 ELSE 0 END +
                CASE WHEN m.glimepiride IN ('Up', 'Down') THEN 1 ELSE 0 END +
                CASE WHEN m.glipizide IN ('Up', 'Down') THEN 1 ELSE 0 END +
                CASE WHEN m.glyburide IN ('Up', 'Down') THEN 1 ELSE 0 END +
                CASE WHEN m.pioglitazone IN ('Up', 'Down') THEN 1 ELSE 0 END +
                CASE WHEN m.rosiglitazone IN ('Up', 'Down') THEN 1 ELSE 0 END +
                CASE WHEN m.insulin IN ('Up', 'Down') THEN 1 ELSE 0 END
            ) as medication_changes,
            
            -- Diabetes medication flag
            MAX(CASE 
                WHEN m.diabetesMed = 'Yes' THEN 1 
                ELSE 0 
            END) as on_diabetes_medication,
            
            -- A1C test result
            MAX(CASE 
                WHEN m.A1Cresult = '>7' THEN 2
                WHEN m.A1Cresult = '>8' THEN 3
                WHEN m.A1Cresult = 'Norm' THEN 1
                ELSE 0
            END) as a1c_test_result
            
        FROM medications m
        GROUP BY m.patient_id, m.encounter_id
        """
        
        logger.info("Extracting medication data...")
        df = pd.read_sql_query(query, self.engine)
        logger.info(f"Extracted {len(df)} medication records")
        return df
    
    def create_master_dataset(self, output_path: str = 'data/raw/patient_data.csv') -> pd.DataFrame:
        """
        Combine all data sources into master dataset
        Performs multiple SQL queries and pandas merges
        """
        logger.info("Creating master dataset from multiple sources...")
        
        # Extract from all sources
        demographics = self.extract_patient_demographics()
        admissions = self.extract_admission_details()
        diagnoses = self.extract_diagnosis_data()
        medications = self.extract_medication_data()
        
        # Merge datasets
        logger.info("Merging datasets...")
        master_df = admissions.merge(demographics, on='patient_id', how='left')
        master_df = master_df.merge(diagnoses, on=['patient_id', 'encounter_id'], how='left')
        master_df = master_df.merge(medications, on=['patient_id', 'encounter_id'], how='left')
        
        # Save to CSV
        master_df.to_csv(output_path, index=False)
        logger.info(f"Master dataset created with {len(master_df)} records")
        logger.info(f"Saved to {output_path}")
        
        return master_df
    
    def get_readmission_statistics(self) -> pd.DataFrame:
        """
        Generate readmission statistics using SQL aggregations
        Demonstrates analytical SQL skills
        """
        query = """
        SELECT 
            CASE 
                WHEN age < 30 THEN '0-29'
                WHEN age BETWEEN 30 AND 49 THEN '30-49'
                WHEN age BETWEEN 50 AND 69 THEN '50-69'
                ELSE '70+'
            END as age_group,
            
            COUNT(*) as total_admissions,
            
            SUM(CASE WHEN readmitted IN ('YES', '<30') THEN 1 ELSE 0 END) as readmissions,
            
            ROUND(100.0 * SUM(CASE WHEN readmitted IN ('YES', '<30') THEN 1 ELSE 0 END) / COUNT(*), 2) as readmission_rate,
            
            AVG(time_in_hospital) as avg_length_of_stay,
            
            AVG(num_medications) as avg_medications,
            
            AVG(num_procedures) as avg_procedures
            
        FROM admissions a
        JOIN patients p ON a.patient_id = p.patient_id
        GROUP BY age_group
        ORDER BY age_group
        """
        
        logger.info("Generating readmission statistics...")
        df = pd.read_sql_query(query, self.engine)
        logger.info("\nReadmission Statistics by Age Group:")
        print(df.to_string(index=False))
        return df
    
    def close_connection(self):
        """Close database connection"""
        self.engine.dispose()
        logger.info("Database connection closed")


if __name__ == "__main__":
    # Example usage
    extractor = PatientDataExtractor()
    
    # Generate statistics
    stats = extractor.get_readmission_statistics()
    
    # Note: In production, you would call create_master_dataset()
    # but this requires actual database tables
    # extractor.create_master_dataset()
    
    extractor.close_connection()