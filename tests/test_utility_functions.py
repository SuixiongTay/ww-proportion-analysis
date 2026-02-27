"""
Unit tests for utility functions in pyEQL_charge_balance.py
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyEQL_charge_balance import citation_rows, median_pH_per_industry, median_TDS_per_industry


class TestCitationRows:
    """Test suite for citation_rows() function"""
    
    def test_citation_rows_basic(self, sample_dataframe):
        """Test basic citation row extraction"""
        result = citation_rows(sample_dataframe)
        
        # Should have 2 citations
        assert len(result) == 2
        assert 'Cite1' in result
        assert 'Cite2' in result
    
    def test_citation_rows_structure(self, sample_dataframe):
        """Test that citation_rows returns correct structure"""
        result = citation_rows(sample_dataframe)
        
        citation = result['Cite1']
        # Should have these keys
        assert 'columns' in citation
        assert 'start_idx' in citation
        assert 'end_idx' in citation
        assert 'element_indices' in citation
        
        # start_idx should be 0
        assert citation['start_idx'] == 0


class TestMedianPH:
    """Test suite for median_pH_per_industry() function"""
    
    def test_median_pH_valid(self, sample_dataframe):
        """Test median pH extraction with valid pH values"""
        result = median_pH_per_industry(sample_dataframe)
        
        # With pH values 7.0 and 6.5, median should be 6.75
        assert result is not None
        assert 6.0 < result < 8.0
    
    def test_median_pH_returns_float(self, sample_dataframe):
        """Test that median pH returns float"""
        result = median_pH_per_industry(sample_dataframe)
        assert isinstance(result, (float, np.floating))


class TestMedianTDS:
    """Test suite for median_TDS_per_industry() function"""
    
    def test_median_TDS_with_data(self):
        """Test TDS extraction with sample data"""
        # Create test DataFrame with TDS values
        data = {
            'Citation': ['Cite1', np.nan, np.nan],
            'Element': ['TDS', 'Na', 'Cl'],
            'Col1': [500.0, 50.0, 70.0],
            'Col2': [480.0, 45.0, 65.0],
        }
        df = pd.DataFrame(data)
        result = median_TDS_per_industry(df)
        
        assert result is not None
        # Median of 500 and 480 is 490
        assert 475 < result < 510
    
    def test_median_TDS_no_data_returns_none(self):
        """Test that TDS with no data returns None"""
        data = {
            'Citation': ['Cite1'],
            'Element': ['Na'],
            'Col1': [50.0],
        }
        df = pd.DataFrame(data)
        result = median_TDS_per_industry(df)
        
        assert result is None
