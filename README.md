# ww-proportion-analysis
WW composition analysis using `pyEQL.equilibrate()`

## Data Flow
- **Input**: Raw csv data from `Cleaned_C_for_pyEQL/` folder 
- **Processing**: `pyEQL_charge_balance.py` - converts csv data to `pyEQL` standardized ions and equilibrates solutions
- **Output**: `pyEQL` equilibrated results are saved to `assets/` folder (JSON and YAML files)
  
## Workflow

![WW Analysis Workflow](assets/workflow.png)
