# include pyEQL engines
from pyEQL.engines import Phreeqc2026EOS
from pyEQL import Solution
from pyEQL.utils import standardize_formula

# other imports
from collections import defaultdict
import pandas as pd
import numpy as np
import pprint

element_to_ion = {
    "Hydrogen": "H[+]",
    "H": "H[+]",
    "Lithium": "Li[+]",
    "Li": "Li[+]",
    "Sodium": "Na[+]",
    "Na": "Na[+]",
    "Potassium": "K[+]",
    "K": "K[+]",
    "Rubidium": "Rb[+]",
    "Rb": "Rb[+]",
    "Cesium": "Cs[+]",
    "Cs": "Cs[+]",
    "Magnesium": "Mg[+2]",
    "Mg": "Mg[+2]",
    "Calcium": "Ca[+2]",
    "Ca": "Ca[+2]",
    "Strontium": "Sr[+2]",
    "Sr": "Sr[+2]",
    "Barium": "Ba[+2]",
    "Ba": "Ba[+2]",
    "Aluminum": "Al[+3]",
    "Al": "Al[+3]",
    "Iron": "Fe[+2]",
    "Fe": "Fe[+2]",
    "Manganese": "Mn[+2]",
    "Mn": "Mn[+2]",
    "Zinc": "Zn[+2]",
    "Zn": "Zn[+2]",
    "Copper": "Cu[+2]",
    "Cu": "Cu[+2]",
    "Nickel": "Ni[+2]",
    "Ni": "Ni[+2]",
    "Cobalt": "Co[+2]",
    "Co": "Co[+2]",
    "Cadmium": "Cd[+2]",
    "Cd": "Cd[+2]",
    "Lead": "Pb[+2]",
    "Pb": "Pb[+2]",
    "Mercury": "Hg[+2]",
    "Hg": "Hg[+2]",
    "Silver": "Ag[+]",
    "Ag": "Ag[+]",
    "Chromium": "Cr[+3]",
    "Cr": "Cr[+3]",
    "Cr+6": "Cr[+6]",
    "Chromium, Trivalent": "Cr[+3]",
    "Chromium, Hexavalent": "CrO4[-2]",
    "Cr+3": "Cr[+3]",
    "Beryllium": "Be[+2]",
    "Beryll": "Be[+2]",
    "Be": "Be[+2]",
    "Scandium": "Sc[+3]",
    "Sc": "Sc[+3]",
    "Titanium": "Ti[+4]",
    "Ti": "Ti[+4]",
    "Vanadium": "V[+3]",
    "V": "V[+3]",
    "Arsenic": "As[+3]",
    "As": "As[+3]",
    "Selenium": "Se[-2]",
    "Se": "Se[-2]",
    "Molybdenum": "Mo[+6]",
    "Mo": "Mo[+6]",
    "Tin": "Sn[+2]",
    "Sn": "Sn[+2]",
    "Antimony": "Sb[+3]",
    "Sb": "Sb[+3]",
    "Thallium": "Tl[+]",
    "Tl": "Tl[+]",
    "Bismuth": "Bi[+3]",
    "Bi": "Bi[+3]",
    "Uranium": "U[+6]",
    "U": "U[+6]",
    "Uranium-238": "U[+6]",
    "Thorium": "Th[+4]",
    "Th": "Th[+4]",
    "Lanthanum": "La[+3]",
    "La": "La[+3]",
    "Cerium": "Ce[+3]",
    "Ce": "Ce[+3]",
    "Neodymium": "Nd[+3]",
    "Nd": "Nd[+3]",
    "Yttrium": "Y[+3]",
    "Y": "Y[+3]",
    "Gallium": "Ga[+3]",
    "Ga": "Ga[+3]",
    "Indium": "In[+3]",
    "In": "In[+3]",
    "Gold": "Au[+3]",
    "Au": "Au[+3]",
    "Platinum": "Pt[+2]",
    "Pt": "Pt[+2]",
    "Palladium": "Pd[+2]",
    "Pd": "Pd[+2]",
    "Zirconium": "Zr[+4]",
    "Zr": "Zr[+4]",
    "Tellurium": "Te[+4]",
    "Te": "Te[+4]",
    "Thiocyanate": "SCN[-]",
    "SCN": "SCN[-]",
    "Tungsten": "W[+6]",
    "W": "W[+6]",
    "Chlorine": "Cl[-]",
    "Chloride": "Cl[-]",
    "Chlorides": "Cl[-]",
    "Cl": "Cl[-]",
    "Chlorate": "ClO3[-]",
    "ClO3": "ClO3[-]",
    "Perchlorate": "ClO4[-]",
    "ClO4": "ClO4[-]",
    "Fluoride": "F[-]",
    "Fluorides": "F[-]",
    "F": "F[-]",
    "Bromide": "Br[-]",
    "Br": "Br[-]",
    "Iodine": "I[-]",
    "I": "I[-]",
    "Sulfate": "SO4[-2]",
    "SO4": "SO4[-2]",
    "Sulfur": "SO4[-2]",
    "S(6.0)": "SO4[-2]",
    "Sulfide": "S[-2]",
    "S": "S[-2]",
    "Ammonia": "NH3(aq)",
    "Ammonia as N": "NH3(aq)",
    "Ammonia as N as N": "NH3(aq)",
    "Ammonia Nitrogen": "NH3(aq)",
    "NH3": "NH3(aq)",
    "Ammonium": "NH4[+]",
    "NH4": "NH4[+]",
    "Nitrate": "NO3[-]",
    "Nitrate as N": "NO3[-]",
    "Nitrate Nitrogen": "NO3[-]",
    "NO3": "NO3[-]",
    "Nitrite": "NO2[-]",
    "Nitrite Nitrogen": "NO2[-]",
    "NO2": "NO2[-]",
    "Cyanide": "CN[-]",
    "CN": "CN[-]",
    "Carbonate": "CO3[-2]",
    "CO3": "CO3[-2]",
    "Bicarbonate ion- (as HCO3)": "HCO3[-]",
    "HCO3": "HCO3[-]",
    "Silicon": "SiO2(aq)",
    "Si": "SiO2(aq)",
    "Silica": "SiO2(aq)",
    "Silicon Dioxide": "SiO2(aq)",
    "CaCO3": "CaCO3(aq)",
    "Boron": "B(OH)3(aq)",
    "B": "B(OH)3(aq)",
    "Nitrogen": "NH4[+]",
    "N": "NH4[+]",
    "Conductivity": None,
    "Specific Conductance @ 25C": None,
    "Temperature": None,
    "°C": None,
    "Total Organic Carbon": None,
    "TOC": None,
    "Total Organic Carbon (Calculated)": None,
    "Total Organic Carbon (mg/L)": None,
    "Total Inorganic Carbon": None,
    "TIC": None,
    "Dissolved Inorganic Carbon": None,
    "DIC": None,
    "Total Dissolved Solids (Calculated)": None,
    "Total Suspended Solids": None,
    "TSS": None,
    "TSS (Reported)": None,
    "Chemical Oxygen Demand": None,
    "COD": None,
    "Chemical Oxygen Demand (COD)": None,
    "Biochemical Oxygen Demand": None,
    "BOD": None,
    "Biochemical Oxygen Demand, 5-day, 20 deg. C": None,
    "BOD5": None,
    "Turbidity": None,
    "NTU": None,
    "Hardness": None,
    "Hardness  Calcium/Magnesium": None,
    "Alkalinity": None,
    "Acidity": None,
    "Acidity as CaCO3": None,
    "Oil": None,
    "Oil and Grease": None,
    "Oil and grease": None,
    "O&G": None,
    "Surfactants": None,
    "Chlorophyll": None,
    "Chl": None,
    "MBAS": None,
    "Methylene Blue Active Substances (MBAS)": None,
    "Residue": None,
    "Concentration (mg/L)": None,
    "Source": None,
    "Element": None,
    "Toluene": None,
    "Benzene": None,
    "Ethylbenzene": None,
    "Xylene": None,
    "o-Xylene": None,
    "Phenol": None,
    "Phenols": None,
    "Phenolics": None,
    "Benzyl Alcohol": None,
    "Benzaldehyde": None,
    "Benzoic Acid": None,
    "Ethanol": None,
    "Isopropyl Alcohol": None,
    "Acetone": None,
    "Hexane": None,
    "Cyclohexanol": None,
    "Naphthalene": None,
    "Phenanthrene": None,
    "Pyrene": None,
    "Fluorene": None,
    "Thorium-228": None,
    "Th-228": None,
    "Thorium-230": None,
    "Th-230": None,
    "Thorium-232": None,
    "Th-232": None,
    "Radium-226": None,
    "Ra-226": None,
    "Radium-228": None,
    "Ra-228": None,
    "Radium-226 and Radium-228": None,
    "Ra-226/Ra-228": None,
}

representative_ions = {
    "Drilling": ["Na[+1]", "Cl[-1]"],  # Na, Cl # needs negative ion - Cl
    "Petroleum Refining": ["Na[+1]", "SO4[-2]"],  # Na, SO4 # needs positive ion - SO4
    "PW Conv": ["Na[+1]", "Cl[-1]"],  # Na, Cl # needs positive ion - Cl
    "PW Unconv": ["Na[+1]", "Cl[-1]"],  # Na, Cl # needs negative ion - Cl
    "Excavation": ["Ca[+2]", "SO4[-2]"],  # Ca, SO4 # needs positive ion - Ca
    "Flotation": ["Na[+1]", "SO4[-2]"],  # Na, SO4 # needs negative ion - SO4
    "Leachate": [
        "Ca[+2]",
        "Al[+3]",
        "Fe[+2]",
        "SO4[-2]",
    ],  # Ca, Al, Fe, SO4 # needs negative ion - SO4
    "Smelting&Refining": ["Ca[+2]", "SO4[-2]"],  # Ca, SO4 # needs negative ion - SO4
    "Gas Scrubber": [
        "Na[+1]",
        "Ca[+2]",
        "SO4[-2]",
    ],  # Na, Ca, SO4 - should be waste gas treatment? # needs positive ion - Na or Ca
    "Tailing": ["Ca[+2]", "SO4[-2]"],  # Ca, SO4 # needs positive ion - Ca
    "Mine Drainage": ["Ca[+2]", "SO4[-2]"],  # Ca, SO4 # needs negative ion - SO4
    "Coal Washing": [
        "Ca[+2]",
        "Mg[+2]",
        "SO4[-2]",
    ],  # Ca, Mg, SO4 # needs positive ion - Ca or Mg
    "Geothermal": ["Na[+1]", "Cl[-1]"],  # Na, Cl # needs negative ion - Cl
    "FGD": [
        "Mg[+2]",
        "Ca[+2]",
        "Cl[-1]",
    ],  # Ca, Mg, Cl # needs positive ion - Ca and Mg
    "Ash": ["Ca[+2]", "SO4[-2]"],  # Ca, SO4 # needs negative ion - SO4
    "CRL": ["Ca[+2]", "SO4[-2]"],  # Ca, SO4 # needs positive ion - Ca
    "Gasification": ["Na[+1]", "SO4[-2]"],  # Na, SO4 # needs negative ion - SO4
    "Semiconductor": [
        "SO4[-2]",
        "Ca[+2]",
        "F[-1]",
    ],  # Ca, F, SO4 # needs positive ion - Ca
    "Tanning": ["Na[+1]", "Cl[-1]"],  # Na, Cl # needs negative ion - Cl
    "Plating": [
        "Na[+1]",
        "Al[+3]",
        "SO4[-2]",
    ],  # Na, Al, SO4 # needs negative ion - SO4
    "Battery Manufacturing": [
        "Na[+1]",
        "K[+1]",
        "SO4[-2]",
    ],  # Na, K, SO4 # needs positive ion - Na or K
    "Battery Recycling": [
        "Na[+1]",
        "F[-1]",
        "SO4[-2]",
    ],  # Na, F, SO4 # needs positive ion - Na
}


# Function to extract each column from the same citation row
def citation_rows(df: pd.DataFrame):
    """
    Extracts citations from DataFrame.
    Groups consecutive rows with the same citation.
    """
    citation_col = df.iloc[:, 0]
    data_col = df.iloc[:, 2:]
    citation_data = {}
    current_idx = 0
    current_citation = None

    for row_idx in range(len(citation_col)):
        citation_name = citation_col.iloc[row_idx]

        if (
            pd.notna(citation_name)
            and citation_name != current_citation
            and current_citation is not None
        ):
            # save citation data
            columns_with_data = []
            for col_idx in range(data_col.shape[1]):
                has_data = False
                for elem_idx in range(current_idx, row_idx):
                    value = data_col.iloc[elem_idx, col_idx]
                    if pd.notna(value):
                        has_data = True
                        break

                if has_data:
                    columns_with_data.append(col_idx)

            citation_data[current_citation] = {
                "columns": columns_with_data,
                "start_idx": current_idx,
                "end_idx": row_idx,
                "element_indices": list(range(current_idx, row_idx)),
            }

            current_idx = row_idx

        if pd.notna(citation_name):
            current_citation = citation_name

    # handle last citation
    if current_citation is not None:
        columns_with_data = []
        for col_idx in range(data_col.shape[1]):
            has_data = False
            for elem_idx in range(current_idx, len(citation_col)):
                value = data_col.iloc[elem_idx, col_idx]
                if pd.notna(value):
                    has_data = True
                    break
            if has_data:
                columns_with_data.append(col_idx)

        citation_data[current_citation] = {
            "columns": columns_with_data,
            "start_idx": current_idx,
            "end_idx": len(citation_col),
            "element_indices": list(range(current_idx, len(citation_col))),
        }

    return citation_data


# Function to extract median pH from the defined industry
def median_pH_per_industry(df: pd.DataFrame):
    """
    Extracts the median pH value for a given industry from the DataFrame.

    Returns:
    float: The median pH value for the specified industry.
    """
    elements_column = df.iloc[:, 1]  # first column
    concentration_column = df.iloc[:, 2:]  # second column onwards

    consolidated_pH = []

    citation_list = citation_rows(df)

    for key, citation in citation_list.items():
        element_idx = citation["element_indices"]
        column_idx = citation["columns"]

        # for col_idx in range(concentration_column.shape[1]): # number of columns
        for ele_idx in element_idx:
            # compile all element names for same citation
            element_name = elements_column[ele_idx]

            for col_idx in column_idx:
                # compile val associated with the element names
                element_value = concentration_column.iloc[ele_idx, col_idx]

                if element_name is not None and element_value is not None:
                    element_name = str(element_name).strip()
                    if element_name in ["ph", "pH", "PH"]:
                        try:
                            pH_value = float(element_value)
                            # print(pH_value)
                            if 0 < pH_value < 14:
                                consolidated_pH.append(pH_value)
                        except Exception:
                            pass

    # pprint.pprint(consolidated_pH)

    median_pH = np.median(consolidated_pH) if consolidated_pH else None

    return median_pH


# Function to extract citation median TDS
def median_TDS_per_industry(df: pd.DataFrame):
    """
    Extracts the median TDS value for a given industry from the DataFrame.

    Returns:
    float: The median TDS value for the specified industry.
    """
    elements_column = df.iloc[:, 1]  # first column
    concentration_column = df.iloc[:, 2:]  # second column onwards

    consolidated_TDS = []

    citation_list = citation_rows(df)

    for key, citation in citation_list.items():
        element_idx = citation["element_indices"]
        column_idx = citation["columns"]

        for ele_idx in element_idx:
            # compile all element names for same citation
            element_name = elements_column[ele_idx]

            for col_idx in column_idx:
                # compile val associated with the element names
                element_value = concentration_column.iloc[ele_idx, col_idx]

                if element_name is not None and element_value is not None:
                    element_name = str(element_name).strip()
                    if element_name in [
                        "tds",
                        "TDS",
                        "total dissolved solids",
                        "Total Dissolved Solids",
                    ]:
                        try:
                            tds_value = float(element_value)
                            if tds_value > 0:
                                consolidated_TDS.append(tds_value)
                        except Exception:
                            pass

    median_TDS = np.median(consolidated_TDS) if consolidated_TDS else None

    return median_TDS


PERCENTAGE_THRESHOLD = 1e-10


class ww_proportion_calculator:
    """
    Calculates the proportion of each ion in a solution based on target charge balance and pH using pyEQL.
    """

    def __init__(self, df, REPRESENTATIVE_IONS, industry, engine="phreeqc2026"):
        # __init__ method
        self.df = df
        self.REPRESENTATIVE_IONS = REPRESENTATIVE_IONS
        self.industry = industry

        # Extract elements and associated concentration values from the DataFrame
        self.elements_col = df.iloc[:, 1].values
        self.concentration_col = df.iloc[:, 2:].values

        # initializing engine
        # self.engine = engine
        self.engine = Phreeqc2026EOS(phreeqc_db="llnl.dat")

        # Extract global pH and tds
        self.global_pH = median_pH_per_industry(df)
        self.global_tds = median_TDS_per_industry(df)

        # defining compositions from citation rows
        self.citation_list = citation_rows(df)

        # median concentration
        self.median_concentrations_from_initial_eq = {}
        self.final_element_proportions_from_initial_median = {}

        # final composition
        self.final_eq = {}
        self.report_equilibrated_proportions = {}

    def _calculated_tds(self, ion_dict_raw: dict):
        """
        Calculates the total dissolved solids (TDS) of the solution based on the ion concentrations.
        """
        calc_tds = sum(ion_dict_raw.values())
        self.calculated_tds = calc_tds
        return self.calculated_tds

    # TODO: can be made to a static method
    def _element_key_to_species(self, element_key: str) -> str:
        """
        Convert () to []
        """
        from pyEQL.utils import standardize_formula

        el, ox = element_key.rstrip(")").split("(")
        el = el.strip()

        if ox.lower() == "unk":
            print(f"unknown oxidation state for '{el}'. Assuming neutral aq species.")
            return f"{el}(aq)"

        # Convert oxidation state to integer
        z = int(float(ox))

        if z == 0:
            return f"{el}(aq)"

        sign = "+" if z > 0 else "-"
        mag = abs(z)
        charge = f"{sign}{mag}" if mag != 1 else sign
        species = f"{el}[{charge}]"

        return standardize_formula(species)

    def _std_element_to_std_ion_dict(self, final_eq_prop: dict, final_pH: float):
        """
        Convert element name to standardized ion name using the element_to_ion mapping.
        """
        median_conc_dict = {}

        for element, conc in final_eq_prop.items():
            # conc = conc_data.get("concentration_mg_L", None)
            # conc = float(conc) if conc is not None else None

            if element in ["O(-2.0)", "H(1.0)", "O(0.0)", "H(0.0)"]:
                continue

            if "(unk)" in str(element) or "unk" in str(
                element
            ):  # S(unk) is present when using llnl.dat
                continue

            if conc is None or conc == 0:
                continue

            if element == "S(6.0)":
                median_conc_dict["SO4[-2]"] = f"{conc * (96.06 / 32.06)} mg/L"

            elif element == "Si(4.0)":
                median_conc_dict["SiO2(aq)"] = f"{conc * (60.08 / 28.09)} mg/L"

            elif element == "N(-3.0)":
                if final_pH < 9.27:
                    median_conc_dict["NH4[+]"] = f"{conc * (18.04 / 14.01)} mg/L"
                else:
                    median_conc_dict["NH4OH(aq)"] = f"{conc * (35.05 / 14.01)} mg/L"

            elif element == "N(3.0)":
                median_conc_dict["NO2[-]"] = f"{conc * (35.05 / 14.01)} mg/L"

            elif element == "N(5.0)":
                median_conc_dict["NO3[-]"] = f"{conc * (62.00 / 14.01)} mg/L"

            elif element == "C(4.0)":
                if final_pH < 6.3:
                    median_conc_dict["H2CO3(aq)"] = f"{conc * (62.03 / 12.01)} mg/L"
                elif 6.3 <= final_pH < 10.3:
                    median_conc_dict["HCO3[-]"] = f"{conc * (61.02 / 12.01)} mg/L"
                else:
                    median_conc_dict["CO3[-2]"] = f"{conc * (60.01 / 12.01)} mg/L"

            elif element == "P(5.0)":
                if final_pH < 1.98:
                    median_conc_dict["H3PO4(aq)"] = f"{conc * (98.00 / 30.97)} mg/L"
                elif 1.98 <= final_pH < 7.19:
                    median_conc_dict["H2PO4[-]"] = f"{conc * (97.00 / 30.97)} mg/L"
                elif 7.19 <= final_pH < 12.03:
                    median_conc_dict["HPO4[-2]"] = f"{conc * (96.00 / 30.97)} mg/L"
                else:
                    median_conc_dict["PO4[-3]"] = f"{conc * (95.00 / 30.97)} mg/L"

            elif element == "B(3.0)":
                if final_pH < 9.24:
                    median_conc_dict["B(OH)3(aq)"] = f"{conc * (61.83 / 10.81)} mg/L"
                elif 9.24 <= final_pH < 12.70:
                    median_conc_dict["H2BO3[-]"] = f"{conc * (62.83 / 10.81)} mg/L"
                elif 12.70 <= final_pH < 13.80:
                    median_conc_dict["HBO3[-2]"] = f"{conc * (57.81 / 10.81)} mg/L"
                else:
                    median_conc_dict["BO3[-3]"] = f"{conc * (56.81 / 10.81)} mg/L"

            else:
                standardized_citation_name = self._element_key_to_species(element)
                median_conc_dict[standardized_citation_name] = f"{conc} mg/L"

        return median_conc_dict

    def _composition_to_proportion_dict(self):
        """
        Converts the final composition of the solution to a dictionary containing their concentration and proportions
        """

        self.final_eq[self.industry] = {}

        for element in self.final_eq_sol.get_components_by_element().keys():
            # remove O and H entries
            if element in ["O(-2.0)", "H(1.0)", "O(0.0)", "H(0.0)"]:
                continue

            if "(unk)" in str(element) or "unk" in str(element):
                print(element)
                continue

            final_val = self.final_eq_sol.get_total_amount(element, "mg/L").magnitude

            if element == "S(6.0)":
                conc_so4 = final_val * (96.06 / 32.06)
                prop_so4 = conc_so4 / self.final_tds * 100.0
                self.final_eq[self.industry]["final_equilibrated_proportions"][
                    "SO4[-2]"
                ] = {
                    "concentration_mg_L": conc_so4,
                    "proportion_percent": prop_so4,
                }

            elif element == "Si(4.0)":
                conc_sio2 = final_val * (60.08 / 28.09)
                prop_sio2 = conc_sio2 / self.final_tds * 100.0
                self.final_eq[self.industry]["final_equilibrated_proportions"][
                    "SiO2(aq)"
                ] = {
                    "concentration_mg_L": conc_sio2,
                    "proportion_percent": prop_sio2,
                }

            elif element == "N(-3.0)":
                if self.final_pH < 9.27:
                    conc_nh4 = final_val * (18.04 / 14.01)
                    prop_nh4 = conc_nh4 / self.final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "NH4[+]"
                    ] = {
                        "concentration_mg_L": conc_nh4,
                        "proportion_percent": prop_nh4,
                    }

                else:
                    conc_no2 = final_val * (35.05 / 14.01)
                    prop_no2 = conc_no2 / self.final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "NH4OH(aq)"
                    ] = {
                        "concentration_mg_L": conc_no2,
                        "proportion_percent": prop_no2,
                    }

            elif element == "N(3.0)":
                conc_no2 = final_val * (46.01 / 14.01)
                prop_no2 = conc_no2 / self.final_tds * 100.0
                self.final_eq[self.industry]["final_equilibrated_proportions"][
                    "NO2[-]"
                ] = {
                    "concentration_mg_L": conc_no2,
                    "proportion_percent": prop_no2,
                }

            elif element == "N(5.0)":
                conc_no3 = final_val * (62.00 / 14.01)
                prop_no3 = conc_no3 / self.final_tds * 100.0
                self.final_eq[self.industry]["final_equilibrated_proportions"][
                    "NO3[-]"
                ] = {
                    "concentration_mg_L": conc_no3,
                    "proportion_percent": prop_no3,
                }

            elif element == "C(4.0)":
                if self.final_pH < 6.3:
                    conc_h2co3 = final_val * (62.03 / 12.01)
                    prop_h2co3 = conc_h2co3 / self.final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "H2CO3(aq)"
                    ] = {
                        "concentration_mg_L": conc_h2co3,
                        "proportion_percent": prop_h2co3,
                    }
                elif 6.3 <= self.final_pH < 10.3:
                    conc_hco3 = final_val * (61.02 / 12.01)
                    prop_hco3 = conc_hco3 / self.final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "HCO3[-]"
                    ] = {
                        "concentration_mg_L": conc_hco3,
                        "proportion_percent": prop_hco3,
                    }
                else:
                    conc_co3 = final_val * (60.01 / 12.01)
                    prop_co3 = conc_co3 / self.final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "CO3[-2]"
                    ] = {
                        "concentration_mg_L": conc_co3,
                        "proportion_percent": prop_co3,
                    }

            elif element == "B(3.0)":
                if self.final_pH < 9.24:
                    conc_boh3 = final_val * (61.83 / 10.81)
                    prop_boh3 = conc_boh3 / self.final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "B(OH)3(aq)"
                    ] = {
                        "concentration_mg_L": conc_boh3,
                        "proportion_percent": prop_boh3,
                    }
                elif 9.24 <= self.final_pH < 12.7:
                    conc_bo3 = final_val * (58.82 / 10.81)
                    prop_bo3 = conc_bo3 / self.final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "H2BO3[-]"
                    ] = {
                        "concentration_mg_L": conc_bo3,
                        "proportion_percent": prop_bo3,
                    }
                elif 12.7 <= self.final_pH < 13.8:
                    conc_bo3_2 = final_val * (57.81 / 10.81)
                    prop_bo3_2 = conc_bo3_2 / self.final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "HBO3[-2]"
                    ] = {
                        "concentration_mg_L": conc_bo3_2,
                        "proportion_percent": prop_bo3_2,
                    }
                else:
                    conc_bo4 = final_val * (56.80 / 10.81)
                    prop_bo4 = conc_bo4 / self.final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "BO4[-3]"
                    ] = {
                        "concentration_mg_L": conc_bo4,
                        "proportion_percent": prop_bo4,
                    }

            else:
                corrected_element = self._element_key_to_species(element)

                self.final_eq[self.industry]["final_equilibrated_proportions"][
                    corrected_element
                ] = {
                    "concentration_mg_L": final_val,
                    "proportion_percent": (final_val / self.final_tds) * 100
                    if self.final_tds > 0
                    else 0,
                }

        return self.final_eq

    def build_ion_data(self):
        """
        Builds the ion data for the solution based on the provided representative ions and their concentrations.
        """
        ion_dict_raw = {}
        seen_ions = set()
        ion_dict_for_sol = {}

        for citation_name, data in self.citation_list.items():
            element_idx = data["element_indices"]
            column_idx = data["columns"]
            ion_dict_raw[citation_name] = {}
            ion_dict_for_sol[citation_name] = {}

            for col_idx in column_idx:
                ion_dict_raw[citation_name][col_idx] = {}
                seen_ions = set()

                for ele_idx in element_idx:
                    # compile all element names for same citation
                    element_name = self.elements_col[ele_idx]
                    element_value = self.concentration_col[ele_idx, col_idx]

                    if element_name is not None and element_value is not None:
                        # to handle nan values
                        if isinstance(element_value, float) and np.isnan(element_value):
                            continue

                        element_name = str(element_name).strip()

                        if element_name in ["ph", "pH", "PH"]:
                            continue

                        if element_name in [
                            "tds",
                            "TDS",
                            "total dissolved solids",
                            "Total Dissolved Solids",
                        ]:
                            continue

                        if element_name in element_to_ion:
                            ion_name = element_to_ion[element_name]
                            # check if column 1 is empty
                            if ion_name is None:
                                continue
                            # handle duplicated ion names
                            if ion_name in seen_ions:
                                continue
                            seen_ions.add(ion_name)

                            try:
                                ion_concentration = float(element_value)
                                if ion_concentration >= 0:
                                    ion_dict_raw[citation_name][col_idx][
                                        ion_name
                                    ] = ion_concentration
                            except Exception:
                                continue

        return ion_dict_raw

    def build_ion_dict(self):
        """
        Builds the ion dict for the solution based on the provided representative ions and their concentrations.
        """
        ion_dict_raw = self.build_ion_data()
        ion_dict_all = {}

        for citation_name, data in ion_dict_raw.items():
            ion_dict_for_sol = {}

            for col_idx, comp in data.items():
                if not comp:
                    continue

                ion_dict_for_sol[col_idx] = {}

                calc_tds = self._calculated_tds(comp)

                for ion_name, conc_val in comp.items():
                    if calc_tds > 0:
                        proportion = (conc_val / calc_tds) * 100

                        if proportion >= PERCENTAGE_THRESHOLD:
                            ion_dict_for_sol[col_idx][ion_name] = f"{conc_val} mg/L"
                        else:
                            ion_dict_for_sol[col_idx][ion_name] = f"{0} mg/L"
                    else:
                        ion_dict_for_sol[col_idx][ion_name] = f"{0} mg/L"

            ion_dict_all[citation_name] = ion_dict_for_sol

        return ion_dict_all

    def build_pH_dict(self):
        """
        Builds the pH dict for the solution.
        """
        pH_dict_for_sol = {}

        for citation_name, data in self.citation_list.items():
            element_idx = data["element_indices"]
            column_idx = data["columns"]

            pH_dict_for_sol[citation_name] = {}

            for col_idx in column_idx:
                found_pH = False

                for ele_idx in element_idx:
                    # compile all element names for same citation
                    element_name = self.elements_col[ele_idx]
                    element_value = self.concentration_col[ele_idx, col_idx]

                    if element_name is not None and element_value is not None:
                        element_name = str(element_name).strip()
                        if element_name in ["ph", "pH", "PH"]:
                            try:
                                pH_value = float(element_value)
                                if 0 < pH_value < 14:
                                    pH_dict_for_sol[citation_name][col_idx] = {
                                        # "column": col_idx,
                                        "pH_value": float(pH_value)
                                    }
                                    found_pH = True
                            except Exception:
                                pass

                if not found_pH:
                    pH_value = float(self.global_pH)
                    if 0 < pH_value < 14:
                        pH_dict_for_sol[citation_name][col_idx] = {
                            # "column": None,
                            "pH_value": float(pH_value)
                        }
                    else:
                        continue

        return pH_dict_for_sol

    def build_solutions(self, ion_dicts: dict, pH_dicts: dict):
        """
        Create all Solution objects once and store them.
        """
        for citation_name, ion_dict in ion_dicts.items():
            self.solutions[citation_name] = {}

            for col_idx, comp in ion_dict.items():
                pH_value = pH_dicts[citation_name][col_idx]["pH_value"]
                # instantiate a new engine
                custom_eos = Phreeqc2026EOS(phreeqc_db="llnl.dat")
                sol = Solution(solutes=comp, pH=pH_value, engine=custom_eos)
                # store solution object
                self.solutions[citation_name][col_idx] = sol

    def build_initial_median_concentration(self):
        """
        Build a initial median concentration for compositions.
        """
        # compile all concentrations from ion_dict_raw
        ion_dict_raw = self.build_ion_data()

        compile_conc_by_element = defaultdict(list)

        for citation_name, data in ion_dict_raw.items():
            for col_idx, comp in data.items():
                for ion_name, conc_val in comp.items():
                    compile_conc_by_element[ion_name].append(conc_val)

        # pprint.pprint(compile_conc_by_element)

        for element, conc_list in compile_conc_by_element.items():
            # calculate median concentration for each element
            median_conc = np.median(conc_list) if conc_list else None
            # compile into instance variable
            self.median_concentrations_from_initial_eq[element] = median_conc

        return self.median_concentrations_from_initial_eq

    def build_final_eq_from_initial_median_concentration(self):
        """
        Build final equilibration results based on initial median concentrations.
        """
        custom_eos = Phreeqc2026EOS(phreeqc_db="llnl.dat")
        initial_median_proportions = self.median_concentrations_from_initial_eq
        initial_median_pH = self.global_pH

        # TODO: can be better written using list comprehension
        ion_dict_for_solution = {}
        for ion, conc in initial_median_proportions.items():
            ion_dict_for_solution[ion] = f"{conc} mg/L"

        initial_sol = Solution(
            solutes=ion_dict_for_solution, pH=initial_median_pH, engine=custom_eos
        )
        initial_chg_bal = initial_sol.charge_balance

        self.final_element_proportions_from_initial_median[self.industry] = {}

        # TODO: can be better written using list comprehension
        for target_ion in self.REPRESENTATIVE_IONS:
            std_target_ion = standardize_formula(target_ion)

            is_negative_ion = "-" in std_target_ion
            is_positive_ion = "+" in std_target_ion

            charge_balance_logic = (initial_chg_bal < 0 and is_positive_ion) or (
                initial_chg_bal > 0 and is_negative_ion
            )

            if not charge_balance_logic:
                continue

            final_sol = None
            equilibrated = False

            ion_dict_for_solution = {
                key: value
                for key, value in ion_dict_for_solution.items()
                if float(value.split()[0]) > 0.0
            }

            # 1st equilibration check
            # If target ion is already present in the initial solution, equilibrate directly
            if std_target_ion in ion_dict_for_solution and charge_balance_logic:
                try:
                    final_sol = Solution(
                        solutes=ion_dict_for_solution,
                        pH=initial_median_pH,
                        balance_charge=std_target_ion,
                        engine=custom_eos,
                    )
                    final_sol.equilibrate()
                    equilibrated = True
                except Exception:
                    pass

            # 2nd equilibration check
            # If target ion is not present, add it with a 1 mg/L concentration, then equilibrate directly
            if not equilibrated and charge_balance_logic:
                comp_w_o_target = ion_dict_for_solution.copy()
                comp_w_o_target[std_target_ion] = "1 mg/L"
                try:
                    final_sol = Solution(
                        solutes=comp_w_o_target,
                        pH=initial_median_pH,
                        balance_charge=std_target_ion,
                        engine=custom_eos,
                    )
                    final_sol.equilibrate()
                    equilibrated = True
                except Exception:
                    pass

            # 3rd equilibration check
            # If all equilibration fails, construct the Solution object with equilibration (but wont speciate)
            if not equilibrated and charge_balance_logic and (final_sol is None):
                try:
                    if std_target_ion in ion_dict_for_solution:
                        final_sol = Solution(
                            solutes=ion_dict_for_solution,
                            pH=initial_median_pH,
                            balance_charge=std_target_ion,
                            engine=custom_eos,
                        )
                        # final_sol.equilibrate()
                        equilibrated = False
                    else:
                        comp_w_o_target = ion_dict_for_solution.copy()
                        comp_w_o_target[std_target_ion] = "1 mg/L"

                        final_sol = Solution(
                            solutes=comp_w_o_target,
                            pH=initial_median_pH,
                            balance_charge=std_target_ion,
                            engine=custom_eos,
                        )
                        # final_sol.equilibrate()
                        equilibrated = False
                except Exception:
                    final_sol = None

            if final_sol is None:
                continue

            # store final_solution in proportions
            final_tds = final_sol.total_dissolved_solids.magnitude
            final_pH = final_sol.pH
            final_charge_balance = final_sol.charge_balance

            self.final_element_proportions_from_initial_median[self.industry][
                std_target_ion
            ] = {
                "solutes": final_sol.as_dict()[
                    "solutes"
                ],  # not working for some reason
                # "solutes": final_sol._solutes,
                "equilibrated": equilibrated,
                "final_pH": final_pH,
                "final_tds_mg_L": final_tds,
                "final_charge_balance": final_charge_balance,
                "proportion_percent": {},
            }

            for element in final_sol.get_components_by_element().keys():
                # remove O and H entries
                if element in ["O(-2.0)", "H(1.0)", "O(0.0)", "H(0.0)"]:
                    continue

                final_val = final_sol.get_total_amount(element, "mg/L").magnitude

                self.final_element_proportions_from_initial_median[self.industry][
                    std_target_ion
                ]["proportion_percent"][element] = {
                    "concentration_mg_L": final_val,
                    "proportion_percent": (final_val / final_tds) * 100,
                }

        # pprint.pprint(self.final_element_proportions_from_initial_median)

        return self.final_element_proportions_from_initial_median

    def build_proportion_from_final_eq_initial_median(self):
        """
        Build final TDS proportions based on final equilibration results from initial median concentrations.
        """
        final_props_from_initial_median = (
            self.final_element_proportions_from_initial_median
        )

        # compile all elements from the final equilibration results
        all_elements = set()
        for industry, std_target_ions in final_props_from_initial_median.items():
            for std_target_ion, target_data in std_target_ions.items():
                prop_data = target_data.get("proportion_percent", {})
                all_elements.update(prop_data.keys())

        element_concentrations = {element: [] for element in all_elements}

        self.report_equilibrated_proportions[self.industry] = {}

        all_pH_values = []
        all_tds_values = []

        for industry, std_target_ions in final_props_from_initial_median.items():
            # print(len(std_target_ions))
            if len(std_target_ions) == 1:
                # make proportions
                for std_target_ion, target_data in std_target_ions.items():
                    # list comprehension to remove solutes with unk
                    solutes = {
                        k: v
                        for k, v in (target_data.get("solutes") or {}).items()
                        if "unk" not in str(k)
                    }
                    pH_val = target_data.get("final_pH", None)

                    pprint.pprint(solutes)

                    self.report_equilibrated_proportions[self.industry] = {
                        "init_dict": solutes,
                        "init_pH": pH_val,
                        "rep_ion_count": len(std_target_ions),
                    }
                return self.report_equilibrated_proportions

            elif len(std_target_ions) > 1:
                # take median concentration
                for std_target_ion, target_data in std_target_ions.items():
                    prop_data = target_data.get("proportion_percent", None)

                    if prop_data is None:
                        print("prop data error")
                        continue

                    # compile pH values for median pH calculation
                    pH_value = target_data.get("final_pH", None)
                    if pH_value is not None:
                        all_pH_values.append(pH_value)
                    # compile tds values for median tds calculation
                    tds_value = target_data.get("final_tds_mg_L", None)
                    if tds_value is not None:
                        all_tds_values.append(tds_value)
                    # compile concentration and proportion for median calculation
                    for element in all_elements:
                        if element in prop_data:
                            conc = prop_data[element].get("concentration_mg_L", None)
                            prop = prop_data[element].get(
                                "proportion_percent", None
                            )  # noqa: F841
                        else:
                            conc = 0.0
                            prop = 0.0  # noqa: F841

                        element_concentrations[element].append(conc)

                median_pH = np.median(all_pH_values) if all_pH_values else None

                temp_dict = {}
                for element in all_elements:
                    conc_list = element_concentrations[element]
                    median_conc = np.median(conc_list) if conc_list else None
                    temp_dict[element] = median_conc

                new_dict = self._std_element_to_std_ion_dict(temp_dict, median_pH)

                self.report_equilibrated_proportions[self.industry] = {
                    "rep_ion_count": len(std_target_ions),
                    "init_pH": median_pH,
                    "init_dict": new_dict,
                }
        # pprint.pprint(self.report_equilibrated_proportions)
        return self.report_equilibrated_proportions

    def final_equilibration_for_initial_median(self):
        """
        Perform a final equilibration step for all solutions to ensure charge balance is achieved.
        """
        from pyEQL import Solution
        from monty.serialization import dumpfn

        custom_eos = Phreeqc2026EOS(phreeqc_db="llnl.dat")

        self.final_eq[self.industry] = {}

        median_conc_dict = self.report_equilibrated_proportions[self.industry].get(
            "init_dict", None
        )
        # pprint.pprint(median_conc_dict)

        for industry, data in self.report_equilibrated_proportions.items():
            # median_conc_dict = data.get("init_dict", None)
            # median_conc_dict = {k: v for k, v in (data.get("init_dict") or {}).items() if "unk" not in str(k).lower() and "unk" not in str(v).lower()}
            median_conc_dict = {
                key: value
                for key, value in median_conc_dict.items()
                if float(value.split()[0]) > 0.0
            }
            # pprint.pprint(median_conc_dict)
            median_pH_dict = data.get("init_pH", None)
            rep_ion_count = data.get("rep_ion_count", None)

            if rep_ion_count is not None and rep_ion_count == 1:
                try:
                    final_eq_sol = Solution(
                        solutes=median_conc_dict, pH=median_pH_dict, engine=custom_eos
                    )
                    # final_eq_sol.equilibrate()
                except Exception as e:
                    return f"failed to equilibrate: {e}"
            elif rep_ion_count is not None and rep_ion_count > 1:
                try:
                    final_eq_sol = Solution(
                        solutes=median_conc_dict,
                        pH=median_pH_dict,
                        balance_charge="auto",
                        engine=custom_eos,
                    )
                    final_eq_sol.equilibrate()
                except Exception as e:
                    return f"failed to equilibrate: {e}"
            else:
                continue

            final_pH = final_eq_sol.pH
            final_tds = final_eq_sol.total_dissolved_solids.magnitude
            final_charge_balance = final_eq_sol.charge_balance

            self.final_eq[self.industry] = {
                "final_eq_pH": final_pH,
                "final_eq_chg_bal": final_charge_balance,
                "final_eq_calc_tds_mg_L": final_tds,
                "reported_median_tds": self.global_tds,
                "final_equilibrated_proportions": {},
            }

            for element in final_eq_sol.get_components_by_element().keys():
                # remove O and H entries
                if element in ["O(-2.0)", "H(1.0)", "O(0.0)", "H(0.0)"]:
                    continue

                if "(unk)" in str(element) or "unk" in str(element):
                    print(element)
                    continue

                final_val = final_eq_sol.get_total_amount(element, "mg/L").magnitude

                if element == "S(6.0)":
                    conc_so4 = final_val * (96.06 / 32.06)
                    prop_so4 = conc_so4 / final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "SO4[-2]"
                    ] = {
                        "concentration_mg_L": conc_so4,
                        "proportion_percent": prop_so4,
                    }

                elif element == "Si(4.0)":
                    conc_sio2 = final_val * (60.08 / 28.09)
                    prop_sio2 = conc_sio2 / final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "SiO2(aq)"
                    ] = {
                        "concentration_mg_L": conc_sio2,
                        "proportion_percent": prop_sio2,
                    }

                elif element == "N(-3.0)":
                    if final_pH < 9.27:
                        conc_nh4 = final_val * (18.04 / 14.01)
                        prop_nh4 = conc_nh4 / final_tds * 100.0
                        self.final_eq[self.industry]["final_equilibrated_proportions"][
                            "NH4[+]"
                        ] = {
                            "concentration_mg_L": conc_nh4,
                            "proportion_percent": prop_nh4,
                        }

                    else:
                        conc_no2 = final_val * (35.05 / 14.01)
                        prop_no2 = conc_no2 / final_tds * 100.0
                        self.final_eq[self.industry]["final_equilibrated_proportions"][
                            "NH4OH(aq)"
                        ] = {
                            "concentration_mg_L": conc_no2,
                            "proportion_percent": prop_no2,
                        }

                elif element == "N(3.0)":
                    conc_no2 = final_val * (46.01 / 14.01)
                    prop_no2 = conc_no2 / final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "NO2[-]"
                    ] = {
                        "concentration_mg_L": conc_no2,
                        "proportion_percent": prop_no2,
                    }

                elif element == "N(5.0)":
                    conc_no3 = final_val * (62.00 / 14.01)
                    prop_no3 = conc_no3 / final_tds * 100.0
                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        "NO3[-]"
                    ] = {
                        "concentration_mg_L": conc_no3,
                        "proportion_percent": prop_no3,
                    }

                elif element == "C(4.0)":
                    if final_pH < 6.3:
                        conc_h2co3 = final_val * (62.03 / 12.01)
                        prop_h2co3 = conc_h2co3 / final_tds * 100.0
                        self.final_eq[self.industry]["final_equilibrated_proportions"][
                            "H2CO3(aq)"
                        ] = {
                            "concentration_mg_L": conc_h2co3,
                            "proportion_percent": prop_h2co3,
                        }
                    elif 6.3 <= final_pH < 10.3:
                        conc_hco3 = final_val * (61.02 / 12.01)
                        prop_hco3 = conc_hco3 / final_tds * 100.0
                        self.final_eq[self.industry]["final_equilibrated_proportions"][
                            "HCO3[-]"
                        ] = {
                            "concentration_mg_L": conc_hco3,
                            "proportion_percent": prop_hco3,
                        }
                    else:
                        conc_co3 = final_val * (60.01 / 12.01)
                        prop_co3 = conc_co3 / final_tds * 100.0
                        self.final_eq[self.industry]["final_equilibrated_proportions"][
                            "CO3[-2]"
                        ] = {
                            "concentration_mg_L": conc_co3,
                            "proportion_percent": prop_co3,
                        }

                elif element == "B(3.0)":
                    if final_pH < 9.24:
                        conc_boh3 = final_val * (61.83 / 10.81)
                        prop_boh3 = conc_boh3 / final_tds * 100.0
                        self.final_eq[self.industry]["final_equilibrated_proportions"][
                            "B(OH)3(aq)"
                        ] = {
                            "concentration_mg_L": conc_boh3,
                            "proportion_percent": prop_boh3,
                        }
                    elif 9.24 <= final_pH < 12.7:
                        conc_bo3 = final_val * (58.82 / 10.81)
                        prop_bo3 = conc_bo3 / final_tds * 100.0
                        self.final_eq[self.industry]["final_equilibrated_proportions"][
                            "H2BO3[-]"
                        ] = {
                            "concentration_mg_L": conc_bo3,
                            "proportion_percent": prop_bo3,
                        }
                    elif 12.7 <= final_pH < 13.8:
                        conc_bo3_2 = final_val * (57.81 / 10.81)
                        prop_bo3_2 = conc_bo3_2 / final_tds * 100.0
                        self.final_eq[self.industry]["final_equilibrated_proportions"][
                            "HBO3[-2]"
                        ] = {
                            "concentration_mg_L": conc_bo3_2,
                            "proportion_percent": prop_bo3_2,
                        }
                    else:
                        conc_bo4 = final_val * (56.80 / 10.81)
                        prop_bo4 = conc_bo4 / final_tds * 100.0
                        self.final_eq[self.industry]["final_equilibrated_proportions"][
                            "BO4[-3]"
                        ] = {
                            "concentration_mg_L": conc_bo4,
                            "proportion_percent": prop_bo4,
                        }

                else:
                    corrected_element = self._element_key_to_species(element)

                    self.final_eq[self.industry]["final_equilibrated_proportions"][
                        corrected_element
                    ] = {
                        "concentration_mg_L": final_val,
                        "proportion_percent": (final_val / final_tds) * 100
                        if final_tds > 0
                        else 0,
                    }

            # manual handing of dict to avoid serialization error
            filename = f"{self.industry}_eq.yaml"
            solution_dict = final_eq_sol.as_dict()
            solution_dict.pop("database", None)
            solution_dict.pop("engine", None)
            dumpfn(solution_dict, "./ww-outputs/" + filename)

        json_name = f"final_eq_{self.industry}.json"

        dumpfn(self.final_eq, "./ww-outputs/" + json_name, indent=2)

        return self.final_eq


INDUSTRY = "Gasification"
raw_data = pd.read_csv(
    f"./Cleaned_C_for_pyEQL/{INDUSTRY}_pyEQL.csv", encoding="windows-1252"
)


# Global function to analyze each citation row and extract ion concentrations
def execution(df: pd.DataFrame, REPRESENTATIVE_IONS: list, industry: str):
    """
    Execute final code
    """

    ww_prop = ww_proportion_calculator(df, REPRESENTATIVE_IONS, industry)
    # calling each function sequentially
    # TODO: refactor to avoid calling function sequentially
    ion_dicts = ww_prop.build_ion_dict()
    initial_median_conc = ww_prop.build_initial_median_concentration()
    prop_from_initial_medians = (
        ww_prop.build_final_eq_from_initial_median_concentration()
    )

    eq_prop_from_initial_medians = (
        ww_prop.build_proportion_from_final_eq_initial_median()
    )

    # final_prop_from_initial_medians = ww_prop.build_median_tds_from_initial_eq_median_proportions()
    final_eq_from_initial_medians = ww_prop.final_equilibration_for_initial_median()

    return (
        ion_dicts,
        initial_median_conc,
        prop_from_initial_medians,
        eq_prop_from_initial_medians,
        final_eq_from_initial_medians,
    )
