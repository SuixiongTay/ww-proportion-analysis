"""
Shared pytest fixtures and configuration for ww-proportion-analysis tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Get the tests directory path
TESTS_DIR = Path(__file__).parent
TEST_DATA_DIR = TESTS_DIR / "test_data"

@pytest.fixture
def sample_ion_dict():
    """Fixture: sample ion dictionary for testing"""
    return {
        'Na[+]': '50.0 mg/L',
        'Ca[+2]': '40.0 mg/L',
        'Cl[-]': '70.0 mg/L',
        'SO4[-2]': '60.0 mg/L',
    }

@pytest.fixture
def sample_pH():
    """Fixture: sample pH value"""
    return 7.0

@pytest.fixture
def sample_element_proportions():
    """Fixture: sample element proportions from pyEQL output"""
    return {
        'Na(1.0)': 23.5,
        'Ca(2.0)': 15.2,
        'Cl(-1.0)': 42.1,
        'S(6.0)': 19.0,
    }

@pytest.fixture
def sample_dataframe():
    """Fixture: minimal sample DataFrame (Battery Manufacturing-like structure)"""
    data = {
        'Citation': ['Cite1', np.nan, np.nan, 'Cite2'],
        'Element': ['pH', 'Na', 'Cl', 'SO4'],
        'Col1': [7.0, 50.0, 70.0, 60.0],
        'Col2': [6.5, 45.0, 65.0, 55.0],
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture: temporary directory for test outputs"""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
