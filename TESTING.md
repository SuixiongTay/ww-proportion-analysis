# Testing guide using pixi for ww-proportion-analysis repo

## Setup with pixi

### 1. Install pixi 
```bash
conda install -c conda-forge pixi
```

### 2. Setup and activate pixi environment
#### note that dev is specific for development environment only
```bash
cd ./ww-proportion-analysis
pixi install
pixi shell --environment dev
```
## Running Tests

### Run all tests:
```bash
pixi run test
```

### Run tests with coverage:
```bash
pixi run test-cov
```
