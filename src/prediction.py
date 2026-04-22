"""
Inference Service for Hospital Readmission Prediction

This module loads a trained model and generates readmission-risk predictions
for single-patient input or batch CSV input.

Author: Jiayi Lyu
"""

import os
import logging
from typing import Dict, Union, Optional

import joblib
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class ReadmissionPredictor:
    """
    Predict readmission risk using a trained classification model.

    The predictor uses the processed training dataset as a feature template
    to ensure inference-time inputs match the model's expected feature space.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pkl",
        template_path: str = "data/processed/cleaned_data.csv",
        target_col: str = "readmitted",
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        logger.info("Loading model from %s", model_path)
        self.model = joblib.load(model_path)
        logger.info("Model loaded successfully")

        logger.info("Loading feature template from %s", template_path)
        template_df = pd.read_csv(template_path)

        if target_col not in template_df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in template file: {template_path}"
            )

        self.target_col = target_col
        self.feature_columns = [col for col in template_df.columns if col != target_col]

        logger.info("Loaded %d feature columns", len(self.feature_columns))

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align incoming data to the exact feature space expected by the model.

        Missing columns are filled with 0.
        Extra columns are ignored.
        Column order is enforced.
        """
        aligned_df = pd.DataFrame(0, index=df.index, columns=self.feature_columns)

        for col in df.columns:
            if col in aligned_df.columns:
                aligned_df[col] = df[col]
            else:
                logger.warning("Ignoring unknown feature: %s", col)

        return aligned_df[self.feature_columns]

    def predict_single_patient(
        self,
        patient_data: Dict[str, Union[int, float]]
    ) -> Dict[str, Union[int, float, str]]:
        """
        Predict readmission risk for a single patient.

        Args:
            patient_data: Dictionary of already-processed or aligned features.

        Returns:
            Dictionary containing class prediction, probability, and risk level.
        """
        input_df = pd.DataFrame([patient_data])
        aligned_df = self._align_features(input_df)

        prediction = int(self.model.predict(aligned_df)[0])
        probability = float(self.model.predict_proba(aligned_df)[0, 1])

        if probability < 0.30:
            risk_level = "Low"
        elif probability < 0.60:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        result = {
            "readmission_prediction": prediction,
            "readmission_probability": probability,
            "risk_level": risk_level,
            "risk_percentage": f"{probability * 100:.1f}%"
        }

        logger.info(
            "Single-patient prediction complete | risk=%s | probability=%.4f",
            risk_level,
            probability,
        )

        return result

    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict readmission risk for multiple patients.

        Args:
            input_df: DataFrame containing one or more rows of patient features.

        Returns:
            DataFrame with prediction columns appended.
        """
        logger.info("Generating predictions for %d patients", len(input_df))

        aligned_df = self._align_features(input_df)

        predictions = self.model.predict(aligned_df)
        probabilities = self.model.predict_proba(aligned_df)[:, 1]

        result_df = input_df.copy()
        result_df["readmission_prediction"] = predictions
        result_df["readmission_probability"] = probabilities
        result_df["risk_level"] = pd.cut(
            probabilities,
            bins=[0.0, 0.3, 0.6, 1.0],
            labels=["Low", "Moderate", "High"],
            include_lowest=True,
        )

        logger.info("Batch prediction complete")
        return result_df

    def predict_from_csv(
        self,
        input_csv: str,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load a CSV file, generate predictions, and optionally save results.

        Args:
            input_csv: Path to CSV file with patient features.
            output_csv: Optional output path for prediction results.

        Returns:
            DataFrame containing predictions.
        """
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        logger.info("Reading input data from %s", input_csv)
        input_df = pd.read_csv(input_csv)

        result_df = self.predict_batch(input_df)

        if output_csv is not None:
            output_dir = os.path.dirname(output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            result_df.to_csv(output_csv, index=False)
            logger.info("Saved predictions to %s", output_csv)

        return result_df

    def get_high_risk_patients(
        self,
        prediction_df: pd.DataFrame,
        threshold: float = 0.60
    ) -> pd.DataFrame:
        """
        Return only patients whose readmission probability exceeds the threshold.
        """
        if "readmission_probability" not in prediction_df.columns:
            raise ValueError(
                "Prediction DataFrame must contain 'readmission_probability'. "
                "Run predict_batch() or predict_from_csv() first."
            )

        high_risk_df = prediction_df[
            prediction_df["readmission_probability"] >= threshold
        ].copy()

        high_risk_df = high_risk_df.sort_values(
            "readmission_probability", ascending=False
        )

        logger.info(
            "Identified %d high-risk patients (threshold=%.2f)",
            len(high_risk_df),
            threshold,
        )

        return high_risk_df


if __name__ == "__main__":
    try:
        predictor = ReadmissionPredictor()

     
        sample_patient = {
            "age": 78,
            "gender": 1,
            "weight": 175,
            "time_in_hospital": 9,
            "num_lab_procedures": 52,
            "num_procedures": 2,
            "num_medications": 20,
            "number_outpatient": 1,
            "number_emergency": 2,
            "number_inpatient": 3,
            "number_diagnoses": 8,
            "total_prior_visits": 6,
            "high_risk_flag": 1,
            "medication_intensity": 2.0,
            "procedure_complexity": 16,
            "is_elderly": 1,
            "is_very_elderly": 1,
            "race_Asian": 0,
            "race_Caucasian": 1,
            "race_Hispanic": 0,
            "race_Other": 0,
            "admission_type_Emergency": 1,
            "admission_type_Trauma": 0,
            "admission_type_Urgent": 0,
        }

        single_result = predictor.predict_single_patient(sample_patient)

        print("\n" + "=" * 60)
        print("SINGLE PATIENT READMISSION RISK ASSESSMENT")
        print("=" * 60)
        print(f"Risk Level: {single_result['risk_level']}")
        print(f"Probability: {single_result['risk_percentage']}")
        print(f"Prediction: {single_result['readmission_prediction']}")
        print("=" * 60)

       
        demo_input = "data/processed/cleaned_data.csv"
        demo_output = "predictions/prediction_results.csv"

        if os.path.exists(demo_input):
            logger.info("Running demo batch prediction using %s", demo_input)

            demo_df = pd.read_csv(demo_input).drop(columns=["readmitted"])
            demo_df = demo_df.head(100)  # portfolio-friendly demo size

            result_df = predictor.predict_batch(demo_df)
            os.makedirs("predictions", exist_ok=True)
            result_df.to_csv(demo_output, index=False)

            logger.info("Saved demo batch predictions to %s", demo_output)

            high_risk_df = predictor.get_high_risk_patients(result_df, threshold=0.60)
            high_risk_path = "predictions/high_risk_patients.csv"
            high_risk_df.to_csv(high_risk_path, index=False)

            logger.info("Saved high-risk patient list to %s", high_risk_path)

            print("\nBatch prediction demo complete.")
            print(f"Saved full predictions to: {demo_output}")
            print(f"Saved high-risk cases to: {high_risk_path}")

    except FileNotFoundError as e:
        print(str(e))
        print("\nPlease run the pipeline first:")
        print("1. python tests\\src\\generate_sample_data.py")
        print("2. python tests\\src\\data_preprocessing.py")
        print("3. python tests\\src\\model_training.py --input data/processed/cleaned_data.csv")
