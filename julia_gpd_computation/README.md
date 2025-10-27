# GPD Form Factors Computation - Julia Package

This package contains Julia code for computing GPD (Generalized Parton Distribution) form factors with high performance.

## Overview

This computation generates:
- **10,000 parameter combinations** (H_alpha, H_beta pairs)
- **Each combination**: 50×50 = 2,500 (xi, t) points  
- **Total computations**: 25,000,000 points
- **Expected runtime**: ~98 hours with 20 threads

## Files

- `main.jl` - Main computation script
- `Project.toml` - Julia package dependencies
- `README.md` - This file

## Setup

### 1. Install Julia
```bash
# On macOS
brew install julia

# On Linux
sudo apt install julia
```

### 2. Install Dependencies
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Usage

### Test Run (10 combinations)
```bash
julia main.jl --test
```

### Full Computation (10,000 combinations)
```bash
julia main.jl
```

## Output

- **CSV files**: One per parameter combination (`alpha_X_beta_Y.csv`)
- **Summary file**: `computation_summary_TIMESTAMP.csv`
- **Each CSV contains**: xi, t, AGFF, DGFF, HCFF_real, HCFF_imag, ECFF_real, ECFF_imag

## Performance

Based on testing:
- **Julia**: 0.281 seconds per point (single-threaded)
- **Python**: 1.325 seconds per point (single-threaded)
- **Julia speedup**: 4.67x faster

## Expected Runtime

| Configuration | Time |
|---------------|------|
| Single-threaded | 98 hours |
| 20 threads | 98 hours (parallel across parameter combinations) |

## Parameters

### Fixed Parameters
- H_norm = 0.5
- H_m = 1.5  
- D1 = -0.5
- mD1 = 2.0
- include_n3 = false

### Variable Parameters
- **H_alpha**: 100 values from -10 to -1
- **H_beta**: 100 values from 3 to 20

### Grid Parameters
- **t range**: -10 to 0 GeV² (50 points)
- **xi range**: 0.3 to 0.8 (50 points)

## Monitoring

The script provides:
- Real-time progress updates
- Runtime estimates
- Summary statistics
- Error handling

## Hardware Requirements

- **CPU**: 20+ cores recommended
- **RAM**: 8+ GB recommended
- **Storage**: 50+ GB for output files

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce number of workers
2. **Slow performance**: Check CPU utilization
3. **Disk space**: Monitor output directory

### Performance Tips

1. **Use SSD storage** for faster I/O
2. **Close other applications** to free CPU
3. **Monitor system resources** during computation

## Results Format

Each output CSV contains:
```csv
xi,t,AGFF,DGFF,HCFF_real,HCFF_imag,ECFF_real,ECFF_imag
0.3,-10.0,0.00309806,-0.0116618,0.00564463,0.0616582,0.0291545,0.0
...
```

## Contact

For questions or issues, please contact the developer.
