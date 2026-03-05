"""
Test for build_ion_dict()
"""

import pandas as pd
from pyEQL_charge_balance import ww_proportion_calculator, representative_ions


# make sure basic functionality is correct
def test_build_ion_dict_leachate():
    """
    Verify build_ion_dict() is behaving as expected for leachate
    """
    INDUSTRY = "Leachate"
    raw_data = pd.read_csv(
        f"./Cleaned_C_for_pyEQL/{INDUSTRY}_pyEQL.csv", encoding="windows-1252"
    )

    # Expected values from a specific citation in Leachate
    expected_ion_dict = {
        "https://www.sciencedirect.com/science/article/pii/S1385894721054474?via%3Dihub": {
            0: {"Mg[+2]": "485.0 mg/L", "SO4[-2]": "17500.0 mg/L"},
            1: {
                "Al[+3]": "950.0 mg/L",
                "Co[+2]": "66.1 mg/L",
                "Cu[+2]": "171.6 mg/L",
                "Fe[+2]": "1541.0 mg/L",
                "Mg[+2]": "2662.0 mg/L",
                "Ni[+2]": "169.0 mg/L",
                "SO4[-2]": "7056.0 mg/L",
                "Zn[+2]": "112.0 mg/L",
            },
        }
    }

    ww_prop = ww_proportion_calculator(
        raw_data, representative_ions[INDUSTRY], INDUSTRY
    )
    actual_ion_dict = ww_prop.build_ion_dict()

    for citation, cols in expected_ion_dict.items():
        assert citation in actual_ion_dict
        actual_cols = actual_ion_dict[citation]

        for col, ion_data in cols.items():
            assert col in actual_cols
            actual_ion_data = actual_cols[col]

            for ion_name, exp_conc in ion_data.items():
                assert ion_name in actual_ion_data
                act_conc = actual_ion_data[ion_name]
                assert exp_conc.strip() == act_conc.strip()


# make sure column 4 is correctly registered
def test_build_ion_dict_excavation():
    """
    Verify build_ion_dict() is behaving as expected for excavation
    """
    INDUSTRY = "Excavation"
    raw_data = pd.read_csv(
        f"./Cleaned_C_for_pyEQL/{INDUSTRY}_pyEQL.csv", encoding="windows-1252"
    )

    expected_ion_dict = {
        "https://journals.scicell.org/index.php/NBC/article/view/417/337": {
            0: {
                "Cu[+2]": "0.2 mg/L",
                "Fe[+2]": "0.05 mg/L",
                "SO4[-2]": "950.0 mg/L",
                "Zn[+2]": "5.6 mg/L",
            },
            4: {
                "Cu[+2]": "1.3 mg/L",
                "Fe[+2]": "205.0 mg/L",
                "SO4[-2]": "1660.0 mg/L",
                "Zn[+2]": "6.9 mg/L",
            },
        }
    }

    ww_prop = ww_proportion_calculator(
        raw_data, representative_ions[INDUSTRY], INDUSTRY
    )
    actual_ion_dict = ww_prop.build_ion_dict()

    for citation, cols in expected_ion_dict.items():
        assert citation in actual_ion_dict
        actual_cols = actual_ion_dict[citation]

        for col, ion_data in cols.items():
            assert col in actual_cols
            actual_ion_data = actual_cols[col]

            for ion_name, exp_conc in ion_data.items():
                assert ion_name in actual_ion_data
                act_conc = actual_ion_data[ion_name]
                assert exp_conc.strip() == act_conc.strip()


# making sure all zeros are correctly registered
def test_build_ion_dict_petroleum():
    """
    Verify build_ion_dict() is behaving as expected for excavation
    """
    INDUSTRY = "Petroleum Refining"
    raw_data = pd.read_csv(
        f"./Cleaned_C_for_pyEQL/{INDUSTRY}_pyEQL.csv", encoding="windows-1252"
    )

    expected_ion_dict = {
        "https://dspace.univ-boumerdes.dz/handle/123456789/11186": {
            0: {"Cd[+2]": "0 mg/L", "Fe[+2]": "0.88 mg/L", "Zn[+2]": "0.03 mg/L"},
            1: {
                "Cd[+2]": "0 mg/L",
                "Cr[+3]": "0 mg/L",
                "Cu[+2]": "0 mg/L",
                "Fe[+2]": "0 mg/L",
                "Pb[+2]": "0 mg/L",
                "Zn[+2]": "0 mg/L",
            },
            2: {
                "Cd[+2]": "0 mg/L",
                "Cr[+3]": "0 mg/L",
                "Cu[+2]": "0 mg/L",
                "Fe[+2]": "0 mg/L",
                "Pb[+2]": "0.14 mg/L",
                "Zn[+2]": "0 mg/L",
            },
        }
    }

    ww_prop = ww_proportion_calculator(
        raw_data, representative_ions[INDUSTRY], INDUSTRY
    )
    actual_ion_dict = ww_prop.build_ion_dict()

    for citation, cols in expected_ion_dict.items():
        assert citation in actual_ion_dict
        actual_cols = actual_ion_dict[citation]

        for col, ion_data in cols.items():
            assert col in actual_cols
            actual_ion_data = actual_cols[col]

            for ion_name, exp_conc in ion_data.items():
                assert ion_name in actual_ion_data
                act_conc = actual_ion_data[ion_name]
                assert exp_conc.strip() == act_conc.strip()
