"""
Basic tests for model functionality
"""

import pytest
import pandas as pd
import numpy as np
from src.prediction import ReadmissionPredictor


def test_sample_data():
    """Test that we can create sample data"""
    data = {
        'age': [65, 70, 55],
        'time_in_hospital': [5, 8, 3]
    }
    df = pd.DataFrame(data)
    assert len(df) == 3
    assert 'age' in df.columns


def test_data_types():
    """Test data type conversions"""
    ages = np.array([65, 70, 55])
    assert ages.dtype == np.int64 or ages.dtype == np.int32


# Add more tests after model training