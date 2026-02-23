from pyEQL import Solution
from pyEQL.utils import standardize_formula
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import pprint

element_to_ion = {
    'Hydrogen': 'H[+]', 'H': 'H[+]',
    'Lithium': 'Li[+]', 'Li': 'Li[+]',
    'Sodium': 'Na[+]', 'Na': 'Na[+]',
    'Potassium': 'K[+]', 'K': 'K[+]',
    'Rubidium': 'Rb[+]', 'Rb': 'Rb[+]',
    'Cesium': 'Cs[+]', 'Cs': 'Cs[+]',
    'Magnesium': 'Mg[+2]', 'Mg': 'Mg[+2]',
    'Calcium': 'Ca[+2]', 'Ca': 'Ca[+2]',
    'Strontium': 'Sr[+2]', 'Sr': 'Sr[+2]',
    'Barium': 'Ba[+2]', 'Ba': 'Ba[+2]',
    'Aluminum': 'Al[+3]', 'Al': 'Al[+3]',
    'Iron': 'Fe[+2]', 'Fe': 'Fe[+2]',
    'Manganese': 'Mn[+2]', 'Mn': 'Mn[+2]',
    'Zinc': 'Zn[+2]', 'Zn': 'Zn[+2]',
    'Copper': 'Cu[+2]', 'Cu': 'Cu[+2]',
    'Nickel': 'Ni[+2]', 'Ni': 'Ni[+2]',
    'Cobalt': 'Co[+2]', 'Co': 'Co[+2]',
    'Cadmium': 'Cd[+2]', 'Cd': 'Cd[+2]',
    'Lead': 'Pb[+2]', 'Pb': 'Pb[+2]',
    'Mercury': 'Hg[+2]', 'Hg': 'Hg[+2]',
    'Silver': 'Ag[+]', 'Ag': 'Ag[+]',
    'Chromium': 'Cr[+3]', 'Cr': 'Cr[+3]', 'Cr+6': 'Cr[+6]',
    'Chromium, Trivalent': 'Cr[+3]',
    'Chromium, Hexavalent': 'CrO4[-2]', 'Cr+3': 'Cr[+3]',
    'Beryllium': 'Be[+2]', 'Beryll': 'Be[+2]', 'Be': 'Be[+2]',
    'Scandium': 'Sc[+3]', 'Sc': 'Sc[+3]',
    'Titanium': 'Ti[+4]', 'Ti': 'Ti[+4]',
    'Vanadium': 'V[+3]', 'V': 'V[+3]',
    'Arsenic': 'As[+3]', 'As': 'As[+3]',
    'Selenium': 'Se[-2]', 'Se': 'Se[-2]',
    'Molybdenum': 'Mo[+6]', 'Mo': 'Mo[+6]',
    'Tin': 'Sn[+2]', 'Sn': 'Sn[+2]',
    'Antimony': 'Sb[+3]', 'Sb': 'Sb[+3]',
    'Thallium': 'Tl[+]', 'Tl': 'Tl[+]',
    'Bismuth': 'Bi[+3]', 'Bi': 'Bi[+3]',
    'Uranium': 'U[+6]', 'U': 'U[+6]',
    'Uranium-238': 'U[+6]',
    'Thorium': 'Th[+4]', 'Th': 'Th[+4]',
    'Lanthanum': 'La[+3]', 'La': 'La[+3]',
    'Cerium': 'Ce[+3]', 'Ce': 'Ce[+3]',
    'Neodymium': 'Nd[+3]', 'Nd': 'Nd[+3]',
    'Yttrium': 'Y[+3]', 'Y': 'Y[+3]',
    'Gallium': 'Ga[+3]', 'Ga': 'Ga[+3]',
    'Indium': 'In[+3]', 'In': 'In[+3]',
    'Gold': 'Au[+3]', 'Au': 'Au[+3]',
    'Platinum': 'Pt[+2]', 'Pt': 'Pt[+2]',
    'Palladium': 'Pd[+2]', 'Pd': 'Pd[+2]',
    'Zirconium': 'Zr[+4]', 'Zr': 'Zr[+4]',
    'Tellurium': 'Te[+4]', 'Te': 'Te[+4]',
    'Thiocyanate': 'SCN[-]', 'SCN': 'SCN[-]',
    'Tungsten': 'W[+6]', 'W': 'W[+6]',
    'Chlorine': 'Cl[-]', 'Chloride': 'Cl[-]', 'Chlorides': 'Cl[-]', 'Cl': 'Cl[-]',
    'Chlorate': 'ClO3[-]', 'ClO3': 'ClO3[-]',
    'Perchlorate': 'ClO4[-]', 'ClO4': 'ClO4[-]',
    'Fluoride': 'F[-]', 'Fluorides': 'F[-]', 'F': 'F[-]',
    'Bromide': 'Br[-]', 'Br': 'Br[-]',
    'Iodine': 'I[-]', 'I': 'I[-]',
    'Sulfate': 'SO4[-2]', 'SO4': 'SO4[-2]', 'Sulfur': 'SO4[-2]', 'S(6.0)': 'SO4[-2]',
    'Sulfide': 'S[-2]', 'S': 'S[-2]',
    'Ammonia': 'NH3(aq)', 'Ammonia as N': 'NH3(aq)', 'Ammonia as N as N': 'NH3(aq)',
    'Ammonia Nitrogen': 'NH3(aq)', 'NH3': 'NH3(aq)',
    'Ammonium': 'NH4[+]', 'NH4': 'NH4[+]',
    'Nitrate': 'NO3[-]', 'Nitrate as N': 'NO3[-]', 'Nitrate Nitrogen': 'NO3[-]', 'NO3': 'NO3[-]',
    'Nitrite': 'NO2[-]', 'Nitrite Nitrogen': 'NO2[-]', 'NO2': 'NO2[-]',
    'Cyanide': 'CN[-]', 'CN': 'CN[-]',
    'Carbonate': 'CO3[-2]', 'CO3': 'CO3[-2]',
    'Bicarbonate ion- (as HCO3)': 'HCO3[-]', 'HCO3': 'HCO3[-]',
    'Silicon': 'SiO2(aq)', 'Si': 'SiO2(aq)', 'Silica': 'SiO2(aq)',
    'Silicon Dioxide': 'SiO2(aq)',
    'CaCO3': 'CaCO3(aq)',
    'Boron': 'B(OH)3(aq)', 'B': 'B(OH)3(aq)',
    'Nitrogen': 'NH4[+]', 'N': 'NH4[+]',

    'Conductivity': None, 'Specific Conductance @ 25C': None,
    'Temperature': None, '°C': None,
    'Total Organic Carbon': None, 'TOC': None,
    'Total Organic Carbon (Calculated)': None, 'Total Organic Carbon (mg/L)': None,
    'Total Inorganic Carbon': None, 'TIC': None, 'Dissolved Inorganic Carbon': None, 'DIC': None,
    'Total Dissolved Solids (Calculated)': None,
    'Total Suspended Solids': None, 'TSS': None, 'TSS (Reported)': None,
    'Chemical Oxygen Demand': None, 'COD': None, 'Chemical Oxygen Demand (COD)': None,
    'Biochemical Oxygen Demand': None, 'BOD': None, 'Biochemical Oxygen Demand, 5-day, 20 deg. C': None, 'BOD5': None,
    'Turbidity': None, 'NTU': None,
    'Hardness': None, 'Hardness  Calcium/Magnesium': None,
    'Alkalinity': None, 'Acidity': None, 'Acidity as CaCO3': None,
    'Oil': None, 'Oil and Grease': None, 'Oil and grease': None, 'O&G': None,
    'Surfactants': None, 'Chlorophyll': None, 'Chl': None,
    'MBAS': None, 'Methylene Blue Active Substances (MBAS)': None,
    'Residue': None,
    'Concentration (mg/L)': None, 'Source': None, 'Element': None,
    'Toluene': None, 'Benzene': None, 'Ethylbenzene': None, 'Xylene': None,
    'o-Xylene': None, 'Phenol': None, 'Phenols': None, 'Phenolics': None,
    'Benzyl Alcohol': None, 'Benzaldehyde': None, 'Benzoic Acid': None,
    'Ethanol': None, 'Isopropyl Alcohol': None, 'Acetone': None,
    'Hexane': None, 'Cyclohexanol': None, 'Naphthalene': None,
    'Phenanthrene': None, 'Pyrene': None, 'Fluorene': None,
    'Thorium-228': None, 'Th-228': None,
    'Thorium-230': None, 'Th-230': None,
    'Thorium-232': None, 'Th-232': None,
    'Radium-226': None, 'Ra-226': None,
    'Radium-228': None, 'Ra-228': None,
    'Radium-226 and Radium-228': None, 'Ra-226/Ra-228': None,
}

representative_ions = {
    'Drilling': ['Na[+]', 'Cl[-]'], # Na, Cl # needs negative ion - Cl
    'Petroleum Refining': ['Na[+]', 'SO4[-2]'], # Na, SO4 # needs positive ion - SO4
    'PW Conv': ['Na[+]', 'Cl[-]'], # Na, Cl # needs positive ion - Cl
    'PW Unconv': ['Na[+]', 'Cl[-]'], # Na, Cl # needs negative ion - Cl
    'Excavation': ['Ca[+2]', 'SO4[-2]'],  # Ca, SO4 # needs positive ion - Ca
    'Flotation': ['Na[+]', 'SO4[-2]'], # Na, SO4 # needs negative ion - SO4
    'Leachate': ['Ca[+2]', 'Al[+3]', 'Fe[+2]', 'SO4[-2]'], # Ca, Al, Fe, SO4 # needs negative ion - SO4
    'Smelting&Refining': ['Ca[+2]', 'SO4[-2]'], # Ca, SO4 # needs negative ion - SO4
    'Gas Scrubber': ['Na[+]', 'Ca[+2]', 'SO4[-2]'], # Na, Ca, SO4 - should be waste gas treatment? # needs positive ion - Na or Ca
    'Tailing': ['Ca[+2]', 'SO4[-2]'], # Ca, SO4 # needs positive ion - Ca
    'Mine Drainage': ['Ca[+2]', 'SO4[-2]'], # Ca, SO4 # needs negative ion - SO4
    'Coal Washing': ['Ca[+2]', 'Mg[+2]', 'SO4[-2]'], # Ca, Mg, SO4 # needs positive ion - Ca or Mg
    'Geothermal': ['Na[+]', 'Cl[-]'], # Na, Cl # needs negative ion - Cl
    'FGD': ['Ca[+2]', 'Mg[+2]', 'Cl[-]'], # Ca, Mg, Cl # needs positive ion - Ca and Mg
    'Ash': ['Ca[+2]', 'SO4[-2]'], # Ca, SO4 # needs negative ion - SO4
    'CRL': ['Ca[+2]', 'SO4[-2]'], # Ca, SO4 # needs positive ion - Ca
    'Gasification': ['Na[+]', 'SO4[-2]'], # Na, SO4 # needs negative ion - SO4 
    'Semiconductor': ['Ca[+2]', 'F[-]', 'SO4[-2]'], # Ca, F, SO4 # needs positive ion - Ca
    'Tanning': ['Na[+]', 'Cl[-]'], # Na, Cl # needs negative ion - Cl
    'Plating': ['Na[+]', 'Al[+3]', 'SO4[-2]'], # Na, Al, SO4 # needs negative ion - SO4
    'Battery Manufacturing': ['Na[+]', 'K[+]', 'SO4[-2]'], # Na, K, SO4 # needs positive ion - Na or K
    'Battery Recycling': ['Na[+]', 'F[-]', 'SO4[-2]'] # Na, F, SO4 # needs positive ion - Na
}

# Function to extract each column from the same citation row
def citation_rows(df: pd.DataFrame):
    """
    Extracts citations from DataFrame.
    
    Returns:
        Dict where each citation has a list of column data
    """
    citation_col = df.iloc[:, 0]
    data_col = df.iloc[:, 2:]
    citation_data = {}
    current_idx = 0
    current_citation = None

    for row_idx in range(len(citation_col)):
        citation_name = citation_col.iloc[row_idx]

        if pd.notna(citation_name):
            if citation_name != current_citation:
                # Save previous citation with ALL columns
                if current_citation is not None:
                    columns_with_data = []
                    for col_idx in range(data_col.shape[1]):
                        # Check if any row in this citation has data in this column
                        has_data = False
                        for elem_idx in range(current_idx, row_idx):
                            value = data_col.iloc[elem_idx, col_idx]
                            if pd.notna(value):
                                has_data = True
                                break
                        
                        if has_data:
                            columns_with_data.append(col_idx)

                    citation_data[current_citation] = {
                        'columns': columns_with_data,  
                        'start_idx': current_idx, 
                        'end_idx': row_idx,
                        'element_indices': list(range(current_idx, row_idx))
                    }
                
                current_citation = citation_name
                current_idx = row_idx

    # Handle last citation
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
            'columns': columns_with_data, 
            'start_idx': current_idx,
            'end_idx': len(citation_col),
            'element_indices': list(range(current_idx, len(citation_col)))
        }

    return citation_data

# Function to extract median pH from the defined industry
def median_pH_per_industry(df: pd.DataFrame):
    """
    Extracts the median pH value for a given industry from the DataFrame.

    Returns:
    float: The median pH value for the specified industry.
    """
    elements_column = df.iloc[:, 1] # first column
    concentration_column = df.iloc[:, 2:] # second column onwards

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
                        except:
                            pass

    median_pH = np.median(consolidated_pH) if consolidated_pH else None

    return median_pH

# Function to extract citation median TDS
def median_TDS_per_industry(df: pd.DataFrame):
    """
    Extracts the median TDS value for a given industry from the DataFrame.

    Returns:
    float: The median TDS value for the specified industry.
    """
    elements_column = df.iloc[:, 1] # first column
    concentration_column = df.iloc[:, 2:] # second column onwards

    consolidated_TDS = []

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
                    if element_name in ["tds", "TDS", "total dissolved solids", "Total Dissolved Solids"]:
                        try:
                            tds_value = float(element_value)
                            # print(tds_value)
                            if tds_value > 0:
                                consolidated_TDS.append(tds_value)
                        except:
                            pass

    median_TDS = np.median(consolidated_TDS) if consolidated_TDS else None

    return median_TDS

PERCENTAGE_THRESHOLD = 1e-3

class ww_proportion_calculator:
    """
    Calculates the proportion of each ion in a solution based on target charge balance and pH using pyEQL.
    """
    def __init__(self, df, REPRESENTATIVE_IONS, industry, engine="phreeqc2026"):
        self.df = df
        self.REPRESENTATIVE_IONS = REPRESENTATIVE_IONS
        self.industry = industry
        # Extract elements and associated concentration values from the DataFrame
        self.elements_col = df.iloc[:, 1].values  
        self.concentration_col = df.iloc[:, 2:].values  
        # pyEQL engine for Solution class
        self.engine = engine
        # Extract global pH and tds
        self.global_pH = median_pH_per_industry(df)
        self.global_tds = median_TDS_per_industry(df)

        self.citation_list = citation_rows(df)
        self.calculated_tds = None
        self.solutions = {}  

        # istance variables for initial and final proportions
        self.initial_element_proportions = None
        self.final_element_proportions = None

        # median proportions of initial and final concentrations and proportions by element
        self.ini_median_conc_by_element = None
        self.ini_median_prop_by_element = None
        self.fin_median_conc_by_element = None
        self.fin_median_prop_by_element = None

        self.final_eq = {}

    def _create_solution(self):
        """
        Creates a pyEQL Solution object based on the provided ion dict and pH.
        """
        self._solution = Solution(engine=self.engine)
        for ion, concentration in self.ion_dict.items():
            standardized_ion = standardize_formula(ion)
            self._solution.add_species(standardized_ion, concentration)
        # instance variable
        self._solution.set_pH(self.pH)

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
            print(f"Warning: Unknown oxidation state for element '{el}'. Assuming neutral species.")
            return f"{el}(aq)"

        # Now safe to convert
        z = int(float(ox))

        # specific ions
        if el == "S":
            if z == 6:
                return "SO4[-2]"
            if z == -2:
                return "S[-2]"  
        elif el == "N":
            if z == 5:
                return "NO3[-]"
            if z == 3:
                return "NO2[-]"
            if z == -3:
                return "NH4[+]"
        elif el == "C":
            if z == 4:
                return "H2CO3(aq)"
        elif el == "P":
            if z == 5:
                return "H3PO4(aq)"
        elif el == "B":
            if z == 3:
                return "B(OH)3(aq)"

        if z == 0:
            return f"{el}(aq)"  

        sign = "+" if z > 0 else "-"
        mag = abs(z)
        charge = f"{sign}{mag}" if mag != 1 else sign
        species = f"{el}[{charge}]"

        return standardize_formula(species) 

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

                        if element_name in ["tds", "TDS", "total dissolved solids", "Total Dissolved Solids"]:
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
                                    ion_dict_raw[citation_name][col_idx][ion_name] = ion_concentration
                            except:
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
                            except:
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
                
                sol = Solution(solutes=comp, pH=pH_value, engine="phreeqc2026")
                # store solution object 
                self.solutions[citation_name][col_idx] = sol
    
    def build_initial_tds_proportion(self):
        """
        Calculate initial TDS proportions using stored solutions.
        """
        if self.initial_element_proportions is None:
            self.initial_element_proportions = {}
        
        ion_dicts = self.build_ion_dict()

        for citation_name, citation_sols in self.solutions.items():
            self.initial_element_proportions[citation_name] = {}
            
            for col_idx, initial_sol in citation_sols.items():  
                self.initial_element_proportions[citation_name][col_idx] = {}
                
                initial_tds = initial_sol.total_dissolved_solids.magnitude
                initial_pH = initial_sol.pH
                
                elements_in_solution = set()
                for element in initial_sol.get_components_by_element().keys():
                    if element in ["O(-2.0)", "H(1.0)", "O(0.0)", "H(0.0)"]:
                        continue
                    
                    ini_val = initial_sol.get_total_amount(element, "mg/L").magnitude

                    if initial_tds > 0:
                        proportion = (ini_val / initial_tds) * 100
                    else:
                        proportion = None
                    
                    self.initial_element_proportions[citation_name][col_idx][element] = {
                        "concentration_mg_L": ini_val,
                        "proportion_percent": proportion,
                        "initial_pH": initial_pH,
                        "initial_tds_mg_L": initial_tds,
                    }
                
                if citation_name in ion_dicts and col_idx in ion_dicts[citation_name]:
                    original_ions = ion_dicts[citation_name][col_idx]
                    
                    for ion_name, conc_str in original_ions.items():
                        # Check if concentration was zero
                        conc_val = float(conc_str.split()[0])
                        
                        if conc_val == 0.0:
                            try:
                                # Use pyEQL to get element info
                                els = initial_sol.get_property(ion_name, "elements")
                                ox = initial_sol.get_property(ion_name, "oxi_state_guesses") or {}
                                
                                for el in els:
                                    if el in ("H", "O"):
                                        continue
                                    if el not in ox:
                                        continue
                                    
                                    element_key = f"{el}({float(ox[el])})"
                                    
                                    # Only add if not already tracked
                                    if element_key not in elements_in_solution:
                                        self.initial_element_proportions[citation_name][col_idx][element_key] = {
                                            "concentration_mg_L": 0.0,
                                            "proportion_percent": 0.0,
                                            "initial_pH": initial_pH,
                                            "initial_tds_mg_L": initial_tds,
                                        }
                            except:
                                pass
        
        return self.initial_element_proportions
    
    def build_final_tds_proportion(self):
        """
        Equilibrate solutions and calculate final TDS proportions.
        """
        # TODO: Need to refactor this code to keep below 200 lines per function

        if self.final_element_proportions is None:
            self.final_element_proportions = {}
        
        for citation_name, citation_sols in self.solutions.items():
            self.final_element_proportions[citation_name] = {}
            
            for col_idx, initial_sol in citation_sols.items():  
                # Get the ion composition from the initial solution
                comp = {species: f"{initial_sol.get_amount(species, 'mg/L').magnitude} mg/L" 
                        for species in initial_sol.components 
                        if species not in ["H2O(aq)", "O2(aq)"]}

                elements_with_zero_conc = set()
                for species, conc_str in comp.items():
                    try:
                        # split val from units and convert to float
                        conc_value = float(conc_str.split()[0]) 
                        if conc_value in [0, 0.0]:
                            # track species in list
                            elements_with_zero_conc.add(species)
                    except:
                        pass
                
                pH_value = initial_sol.pH
                final_chg_bal = initial_sol.charge_balance
                
                # Try equilibration              
                for target_ion in self.REPRESENTATIVE_IONS:
                    # standardized target ion formula
                    std_target_ion = standardize_formula(target_ion)

                    if std_target_ion not in self.final_element_proportions[citation_name]:
                        self.final_element_proportions[citation_name][std_target_ion] = {}

                    is_negative_ion = "-" in target_ion
                    is_positive_ion = "+" in target_ion
                    
                    # Define charge balance logic
                    charge_balance_logic = (
                        (final_chg_bal < 0 and is_positive_ion) or 
                        (final_chg_bal > 0 and is_negative_ion)
                    )
                    
                    if not charge_balance_logic:
                        continue

                    final_sol = None
                    equilibrated = False
                    # 1st equilibration check
                    # If target ion is already present in the initial solution, equilibrate directly
                    if target_ion in comp and charge_balance_logic:
                        try:
                            final_sol = Solution(
                                solutes=comp, 
                                pH=pH_value, 
                                balance_charge=target_ion, 
                                engine="phreeqc2026"
                            )
                            final_sol.equilibrate()
                            equilibrated = True
                        except:
                            pass

                    # 2nd equilibration check
                    # If target ion is not present, add it with a 1 mg/L concentration, then equilibrate directly
                    if not equilibrated and charge_balance_logic:
                        comp_w_o_target = comp.copy()
                        comp_w_o_target[std_target_ion] = "1 mg/L"
                        try:
                            final_sol = Solution(
                                solutes=comp_w_o_target, 
                                pH=pH_value, 
                                balance_charge=std_target_ion, 
                                engine="phreeqc2026"
                            )
                            final_sol.equilibrate()
                            equilibrated = True
                        except:
                            pass

                    # 3rd equilibration check
                    # If all equilibration fails, construct the Solution object with equilibration (but wont speciate)
                    if not equilibrated and charge_balance_logic and (final_sol is None):
                        try:
                            if target_ion in comp:
                                final_sol = Solution(
                                    solutes=comp, 
                                    pH=pH_value, 
                                    balance_charge=target_ion, 
                                    engine="phreeqc2026"
                                )
                                # final_sol.equilibrate()
                                equilibrated = False
                            else:
                                comp_w_o_target = comp.copy()
                                comp_w_o_target[std_target_ion] = "1 mg/L"
                                
                                final_sol = Solution(
                                    solutes=comp_w_o_target, 
                                    pH=pH_value, 
                                    balance_charge=std_target_ion, 
                                    engine="phreeqc2026"
                                )
                                # final_sol.equilibrate()
                                equilibrated = False
                        except:
                            final_sol = None
                    
                    if final_sol is None:
                        continue

                    # store final_solution in proportions
                    self.final_element_proportions[citation_name][std_target_ion][col_idx] = {}
                    final_tds = final_sol.total_dissolved_solids.magnitude
                    
                    for element in final_sol.get_components_by_element().keys():
                        # remove O and H entries
                        if element in ["O(-2.0)", "H(1.0)", "O(0.0)", "H(0.0)"]:
                            continue
                        
                        final_val = final_sol.get_total_amount(element, "mg/L").magnitude
                        final_charge_balance = final_sol.charge_balance
                        final_pH = final_sol.pH
                        
                        # self.final_element_proportions[citation_name][std_target_ion][col_idx][element] = {
                        #     "concentration_mg_L": final_val,
                        #     "proportion_percent": (final_val / final_tds) * 100,
                        #     "final_tds_mg_L": final_tds,
                        #     "final_pH": final_pH,
                        #     "equilibrated": equilibrated,
                        #     "charge_balance": final_charge_balance,
                        # }

                        if equilibrated:
                            self.final_element_proportions[citation_name][std_target_ion][col_idx][element] = {
                                "concentration_mg_L": final_val,
                                "proportion_percent": (final_val / final_tds) * 100,
                                "final_tds_mg_L": final_tds,
                                "final_pH": final_pH,
                                "equilibrated": equilibrated,
                                "charge_balance": final_charge_balance,
                            }
                    
                    for zero_element in elements_with_zero_conc:
                        #zero element is an ion
                        try:
                            els = final_sol.get_property(zero_element, "elements")
                            ox  = final_sol.get_property(zero_element, "oxi_state_guesses") or {}
                        except Exception:
                            continue

                        for el in els:
                            if el in ("H", "O"):
                                continue
                            if el not in ox:
                                continue

                            element_key = f"{el}({float(ox[el])})"

                        # # Check if this element is already in the results
                        # self.final_element_proportions[citation_name][std_target_ion][col_idx][element_key] = {
                        #     "concentration_mg_L": 0.0,
                        #     "proportion_percent": 0.0,
                        #     "final_tds_mg_L": final_tds,
                        #     "final_pH": final_pH,
                        #     "equilibrated": equilibrated,
                        #     "charge_balance": final_charge_balance,
                        # }

                        if equilibrated:
                            self.final_element_proportions[citation_name][std_target_ion][col_idx][element_key] = {
                            "concentration_mg_L": 0.0,
                            "proportion_percent": 0.0,
                            "final_tds_mg_L": final_tds,
                            "final_pH": final_pH,
                            "equilibrated": equilibrated,
                            "charge_balance": final_charge_balance,
                        }

        return self.final_element_proportions

    def build_proportions_data_dict(self):
        """
        Build a comprehensive data dictionary containing all initial and final proportions for all citations and representative ions.
        """
        import numpy as np
        # first compile initial proportions data
        compile_initial_median_concentrations_by_element = defaultdict(list)
        compile_initial_median_proportions_by_element = defaultdict(list)
        compile_initial_tds = []
        compile_initial_pH = []

        self.ini_median_conc_by_element = defaultdict(list)
        self.ini_median_prop_by_element = defaultdict(list)
        
        if self.initial_element_proportions:
            for citation_name, col_data in self.initial_element_proportions.items():
                for col_idx, element_data in col_data.items():
                    # get initial tds for each column
                    tds_value = None
                    pH_value = None
                    for data in element_data.values():
                        if data and data.get("initial_tds_mg_L") is not None:
                            tds_value = data["initial_tds_mg_L"]
                        if data and data.get("initial_pH") is not None:
                            pH_value = data["initial_pH"]
                            break

                    if tds_value is not None:
                        compile_initial_tds.append(float(tds_value))
                    if pH_value is not None:
                        compile_initial_pH.append(float(pH_value))
                    # get initial concentration and proportion for each element in the column
                    for element, data in element_data.items():
                        # initial_median_concentrations_by_element[element].append(data["concentration_mg_L"])
                        # initial_median_proportions_by_element[element].append(data["proportion_percent"])
                        initial_conc = data.get("concentration_mg_L", None)
                        initial_prop = data.get("proportion_percent", None)
                        if initial_conc is not None:
                            compile_initial_median_concentrations_by_element[element].append(initial_conc)
                        if initial_prop is not None:
                            compile_initial_median_proportions_by_element[element].append(initial_prop)

        for element, conc_list in compile_initial_median_concentrations_by_element.items():
            median_conc = np.median(conc_list) if conc_list else None
            self.ini_median_conc_by_element[element] = median_conc
        for element, prop_list in compile_initial_median_proportions_by_element.items():
            median_prop = np.median(prop_list) if conc_list else None
            self.ini_median_prop_by_element[element] = median_prop
        
        self.median_initial_pH = np.median(compile_initial_pH) if compile_initial_pH else None
        self.median_initial_tds = np.median(compile_initial_tds) if compile_initial_tds else None

        # first compile final proportions data
        compile_final_median_concentrations_by_element = defaultdict(list)
        compile_final_median_proportions_by_element = defaultdict(list)
        compile_final_tds = []
        compile_final_pH = []

        self.fin_median_conc_by_element = defaultdict(list)
        self.fin_median_prop_by_element = defaultdict(list)
        
        if self.final_element_proportions:
            for citation_name, target_data in self.final_element_proportions.items():
                for target_ion, col_data in target_data.items():
                    for col_idx, element_data in col_data.items():
                        # get final tds for each column
                        tds_value = None
                        pH_value = None
                        for data in element_data.values():
                            if data and data.get("final_tds_mg_L") is not None:
                                tds_value = data["final_tds_mg_L"]
                            if data and data.get("final_pH") is not None:
                                pH_value = data["final_pH"]
                                break

                        if tds_value is not None:
                            compile_final_tds.append(float(tds_value))
                        if pH_value is not None:
                            compile_final_pH.append(float(pH_value))
                        # get final concentration and proportion for each element in the column
                        for element, data in element_data.items():
                            # final_median_concentrations_by_element[element].append(data["concentration_mg_L"])
                            # final_median_proportions_by_element[element].append(data["proportion_percent"])
                            final_conc = data.get("concentration_mg_L", None)
                            final_prop = data.get("proportion_percent", None)
                            if final_conc is not None:
                                compile_final_median_concentrations_by_element[element].append(final_conc)
                            if final_prop is not None:
                                compile_final_median_proportions_by_element[element].append(final_prop)
        
        for element, conc_list in compile_final_median_concentrations_by_element.items():
            median_conc = np.median(conc_list) if conc_list else None
            self.fin_median_conc_by_element[element] = median_conc
        for element, prop_list in compile_final_median_proportions_by_element.items():
            median_prop = np.median(prop_list) if conc_list else None
            self.fin_median_prop_by_element[element] = median_prop
        
        self.median_final_pH = np.median(compile_final_pH) if compile_final_pH else None
        self.median_final_tds = np.median(compile_final_tds) if compile_final_tds else None

        return self.ini_median_conc_by_element, self.fin_median_conc_by_element, self.ini_median_prop_by_element, self.fin_median_prop_by_element, self.median_initial_pH, self.median_initial_tds, self.median_final_pH, self.median_final_tds, compile_initial_median_concentrations_by_element, compile_final_median_concentrations_by_element, compile_initial_median_proportions_by_element, compile_final_median_proportions_by_element

    def final_equilibration_step(self):
        """
        Perform a final equilibration step for all solutions to ensure charge balance is achieved.
        """
        from pyEQL import Solution
        from monty.serialization import dumpfn, loadfn

        median_conc_dict = {}
        # take final solution and create a solution
        for citation_name, citation_sols in self.fin_median_conc_by_element.items():

            if "unk" in str(citation_name):
                print(f"{citation_name}")
                # continue
                
            standardized_citation_name = self._element_key_to_species(citation_name)
            if citation_sols > 0:
                median_conc_dict[standardized_citation_name] = f"{citation_sols} mg/L"
            else:
                pass

        if self.industry not in self.final_eq:
            self.final_eq[self.industry] = {}

        final_eq_sol = Solution(solutes=median_conc_dict, pH=self.median_final_pH, balance_charge="auto", engine="phreeqc2026")
        final_eq_sol.equilibrate()

        final_pH = final_eq_sol.pH
        final_tds = final_eq_sol.total_dissolved_solids.magnitude
        final_charge_balance = final_eq_sol.charge_balance

        self.final_eq[self.industry] = {
                    "final_eq_pH": final_pH,
                    "final_eq_chg_bal": final_charge_balance,
                    "final_eq_calc_tds_mg_L": final_tds,
                    "reported_median_tds": self.global_tds,
                    "final_equilibrated_proportions": {}
                }

        for element in final_eq_sol.get_components_by_element().keys():
            # remove O and H entries
            if element in ["O(-2.0)", "H(1.0)", "O(0.0)", "H(0.0)"]:
                continue
            
            if "(unk)" in str(element) or "unk" in str(element):
                print(element)
                # continue
            
            final_val = final_eq_sol.get_total_amount(element, "mg/L").magnitude

            if element == "S(6.0)":

                # prop_s = (final_val / final_tds * 100.0) 
                # self.final_eq[self.industry]["final_equilibrated_proportions"]["S(6.0)"] = {
                #     "concentration_mg_L": final_val,
                #     "proportion_percent": prop_s,
                # }

                conc_so4 = final_val * (96.06 / 32.06)
                prop_so4 = (conc_so4 / final_tds * 100.0) 
                self.final_eq[self.industry]["final_equilibrated_proportions"]["SO4[-2]"] = {
                    "concentration_mg_L": conc_so4,
                    "proportion_percent": prop_so4,
                }
            
            elif element == "Si(4.0)":
                # prop_si = (final_val / final_tds * 100.0) 
                # self.final_eq[self.industry]["final_equilibrated_proportions"]["Si(4.0)"] = {
                #     "concentration_mg_L": final_val,
                #     "proportion_percent": prop_si,
                # }

                conc_sio2 = final_val * (60.08 / 28.09)
                prop_sio2 = (conc_sio2 / final_tds * 100.0) 
                self.final_eq[self.industry]["final_equilibrated_proportions"]["SiO2(aq)"] = {
                    "concentration_mg_L": conc_sio2,
                    "proportion_percent": prop_sio2,
                }

            elif element == "N(-3.0)":

                # prop_n = (final_val / final_tds * 100.0) 
                # self.final_eq[self.industry]["final_equilibrated_proportions"]["N(-3.0)"] = {
                #     "concentration_mg_L": final_val,
                #     "proportion_percent": prop_n,
                # }
                if final_pH < 9.27:

                    conc_nh4 = final_val * (18.04 / 14.01)
                    prop_nh4 = (conc_nh4 / final_tds * 100.0) 
                    self.final_eq[self.industry]["final_equilibrated_proportions"]["NH4[+]"] = {
                        "concentration_mg_L": conc_nh4,
                        "proportion_percent": prop_nh4,
                    }
                
                else:

                    conc_no2 = final_val * (35.05 / 14.01)
                    prop_no2 = (conc_no2 / final_tds * 100.0) 
                    self.final_eq[self.industry]["final_equilibrated_proportions"]["NH4OH(aq)"] = {
                        "concentration_mg_L": conc_no2,
                        "proportion_percent": prop_no2,
                    }
                
            elif element == "N(3.0)":

                # prop_n = (final_val / final_tds * 100.0) 
                # self.final_eq[self.industry]["final_equilibrated_proportions"]["N(3.0)"] = {
                #     "concentration_mg_L": final_val,
                #     "proportion_percent": prop_n,
                # }

                conc_no2 = final_val * (35.05 / 14.01)
                prop_no2 = (conc_no2 / final_tds * 100.0) 
                self.final_eq[self.industry]["final_equilibrated_proportions"]["NO2[-]"] = {
                    "concentration_mg_L": conc_no2,
                    "proportion_percent": prop_no2,
                }

            elif element == "N(5.0)":

                # prop_n = (final_val / final_tds * 100.0) 
                # self.final_eq[self.industry]["final_equilibrated_proportions"]["N(5.0)"] = {
                #     "concentration_mg_L": final_val,
                #     "proportion_percent": prop_n,
                # }

                conc_no3 = final_val * (62.00 / 14.01)
                prop_no3 = (conc_no3 / final_tds * 100.0) 
                self.final_eq[self.industry]["final_equilibrated_proportions"]["NO3[-]"] = {
                    "concentration_mg_L": conc_no3,
                    "proportion_percent": prop_no3,
                }

            elif element == "C(4.0)":

                # prop_c = (final_val / final_tds * 100.0) 
                # self.final_eq[self.industry]["final_equilibrated_proportions"]["C(4.0)"] = {
                #     "concentration_mg_L": final_val,
                #     "proportion_percent": prop_c,
                # }
                if final_pH < 6.3:
                    conc_h2co3 = final_val * (62.03 / 12.01)
                    prop_h2co3 = (conc_h2co3 / final_tds * 100.0) 
                    self.final_eq[self.industry]["final_equilibrated_proportions"]["H2CO3(aq)"] = {
                        "concentration_mg_L": conc_h2co3,
                        "proportion_percent": prop_h2co3,
                    }
                elif 6.3 <= final_pH < 10.3:
                    conc_hco3 = final_val * (61.02 / 12.01)
                    prop_hco3 = (conc_hco3 / final_tds * 100.0) 
                    self.final_eq[self.industry]["final_equilibrated_proportions"]["HCO3[-]"] = {
                        "concentration_mg_L": conc_hco3,
                        "proportion_percent": prop_hco3,
                    }
                else:
                    conc_co3 = final_val * (60.01 / 12.01)
                    prop_co3 = (conc_co3 / final_tds * 100.0) 
                    self.final_eq[self.industry]["final_equilibrated_proportions"]["CO3[-2]"] = {
                        "concentration_mg_L": conc_co3,
                        "proportion_percent": prop_co3,
                    }

            elif element == "B(3.0)":

                # prop_b = (final_val / final_tds * 100.0) 
                # self.final_eq[self.industry]["final_equilibrated_proportions"]["B(3.0)"] = {
                #     "concentration_mg_L": final_val,
                #     "proportion_percent": prop_b,
                # }

                if final_pH < 9.24:
                    conc_boh3 = final_val * (61.83 / 10.81)
                    prop_boh3 = (conc_boh3 / final_tds * 100.0) 
                    self.final_eq[self.industry]["final_equilibrated_proportions"]["B(OH)3(aq)"] = {
                        "concentration_mg_L": conc_boh3,
                        "proportion_percent": prop_boh3,
                    }
                if 9.24 <= final_pH < 12.7:
                    conc_bo3 = final_val * (58.82 / 10.81)
                    prop_bo3 = (conc_bo3 / final_tds * 100.0)
                    self.final_eq[self.industry]["final_equilibrated_proportions"]["H2BO3[-]"] = {
                        "concentration_mg_L": conc_bo3,
                        "proportion_percent": (prop_bo3 / final_tds) * 100.0,
                    }
                if 12.7 <= final_pH < 13.8:
                    conc_bo3_2 = final_val * (57.81 / 10.81)
                    prop_bo3_2 = (conc_bo3_2 / final_tds * 100.0)
                    self.final_eq[self.industry]["final_equilibrated_proportions"]["HBO3[-2]"] = {
                        "concentration_mg_L": conc_bo3_2,
                        "proportion_percent": prop_bo3_2,
                    }
                else:
                    conc_bo4 = final_val * (56.80 / 10.81)
                    prop_bo4 = (conc_bo4 / final_tds * 100.0)
                    self.final_eq[self.industry]["final_equilibrated_proportions"]["BO4[-3]"] = {
                        "concentration_mg_L": conc_bo4,
                        "proportion_percent": prop_bo4,
                    }
            
            else:
            
                corrected_element = self._element_key_to_species(element)
                
                self.final_eq[self.industry]["final_equilibrated_proportions"][corrected_element] = {
                    "concentration_mg_L": final_val,
                    "proportion_percent": (final_val / final_tds) * 100 if final_tds > 0 else 0,
                }

        # sol to yaml
        filename = f"{self.industry}_eq.yaml"
        final_eq_sol.to_file("./final_eq_solutions/" + filename)

        json_name = f"final_eq_{self.industry}.json"

        dumpfn(self.final_eq, "./final_eq_solutions/" + json_name, indent=2)
        
        return self.final_eq

INDUSTRY = "Excavation"
raw_data = pd.read_csv(f"./Cleaned_C_for_pyEQL/{INDUSTRY}_pyEQL.csv", encoding="windows-1252")

# Global function to analyze each citation row and extract ion concentrations
def execution(df: pd.DataFrame, REPRESENTATIVE_IONS: list, industry: str):
    """
    Execute final code 
    """
    from pyEQL import Solution

    ww_prop = ww_proportion_calculator(df, REPRESENTATIVE_IONS, industry)
    # calling each function sequentially
    # TODO: refactor to avoid calling function sequentially 
    ion_dicts = ww_prop.build_ion_dict()
    pH_dicts = ww_prop.build_pH_dict()
    build_sols = ww_prop.build_solutions(ion_dicts, pH_dicts)
    initial_tds_proportions = ww_prop.build_initial_tds_proportion()
    final_tds_proportions = ww_prop.build_final_tds_proportion()

    prop_dict = ww_prop.build_proportions_data_dict()
    final_prop = ww_prop.final_equilibration_step()

    return ion_dicts, initial_tds_proportions, final_tds_proportions, prop_dict, final_prop

import os
folder_path = "./final_eq_solutions"
file_name = f"{INDUSTRY}.txt"
full_path = os.path.join(folder_path, file_name)

data = execution(raw_data, representative_ions[INDUSTRY], INDUSTRY)

with open(full_path, "w") as f:
    pprint.pprint(data, stream=f)