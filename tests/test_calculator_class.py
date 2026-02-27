"""
Unit tests for ww_proportion_calculator class
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyEQL_charge_balance import ww_proportion_calculator


class TestWWProportionCalculator:
    """Test suite for ww_proportion_calculator class"""
    
    @pytest.fixture
    def sample_calculator(self, sample_dataframe):
        """Create a sample calculator instance"""
        representative_ions = ['Na[+1]', 'Cl[-1]']
        calc = ww_proportion_calculator(sample_dataframe, representative_ions, 'TestIndustry')
        return calc
    
    def test_calculator_initialization(self, sample_calculator):
        """Test that calculator initializes properly"""
        assert sample_calculator.industry == 'TestIndustry'
        assert sample_calculator.median_concentrations_from_initial_eq == {}
        assert sample_calculator.final_element_proportions_from_initial_median == {}
    
    def test_calculator_has_engine(self, sample_calculator):
        """Test that calculator has Phreeqc2026EOS engine"""
        assert sample_calculator.engine is not None
        # Engine should have phreeqc_db attribute
        assert hasattr(sample_calculator.engine, 'phreeqc_db')
    
    def test_build_ion_data(self, sample_calculator):
        """Test build_ion_data method"""
        ion_data = sample_calculator.build_ion_data()
        
        # Should return a dict
        assert isinstance(ion_data, dict)
        # Should have at least one citation
        assert len(ion_data) > 0
    
    def test_build_initial_median_concentration(self, sample_calculator):
        """Test build_initial_median_concentration method"""
        result = sample_calculator.build_initial_median_concentration()
        
        assert isinstance(result, dict)
        # Result should be in the instance variable
        assert sample_calculator.median_concentrations_from_initial_eq == result
    
    def test_calculated_tds_sums_correctly(self, sample_calculator, sample_ion_dict):
        """Test _calculated_tds method"""
        # Create a dict with numeric values (not strings with units)
        ion_dict = {'Na[+]': 50.0, 'Ca[+2]': 40.0, 'Cl[-]': 70.0}
        
        tds = sample_calculator._calculated_tds(ion_dict)
        
        assert tds == pytest.approx(160.0)
    
    def test_element_key_to_species_conversion(self, sample_calculator):
        """Test _element_key_to_species method"""
        # Test conversion of element notation to species
        result = sample_calculator._element_key_to_species("Na(1.0)")
        
        assert result is not None
        # Should contain Na
        assert "Na" in result
    
    @patch('pyEQL_charge_balance.Solution')
    def test_build_ion_dict_structure(self, mock_solution, sample_calculator):
        """Test that build_ion_dict returns properly formatted dict"""
        result = sample_calculator.build_ion_dict()
        
        assert isinstance(result, dict)
        # Each entry should map to a dict
        for citation, data in result.items():
            assert isinstance(data, dict)


class TestCalculatorDataStorage:
    """Test data storage patterns in calculator"""
    
    @pytest.fixture
    def sample_calculator(self, sample_dataframe):
        representative_ions = ['Na[+1]', 'Cl[-1]']
        return ww_proportion_calculator(sample_dataframe, representative_ions, 'TestIndustry')
    
    def test_report_equilibrated_proportions_initialized(self, sample_calculator):
        """Test that report_equilibrated_proportions is properly initialized"""
        assert isinstance(sample_calculator.report_equilibrated_proportions, dict)
        assert len(sample_calculator.report_equilibrated_proportions) == 0
    
    def test_final_element_proportions_initialized(self, sample_calculator):
        """Test that final_element_proportions_from_initial_median is initialized"""
        assert isinstance(sample_calculator.final_element_proportions_from_initial_median, dict)
        assert len(sample_calculator.final_element_proportions_from_initial_median) == 0
