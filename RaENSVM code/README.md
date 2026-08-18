# RaENSVM

This directory provides the R implementation and usage examples for the Rescaled asymmetric Elastic Net Support Vector Machine (RaENSVM).

## Repository scope

The implementation was originally developed using utility functions from the [`manysvms`](https://github.com/define957/manysvms) R package. To facilitate its use, the kernel, prediction, evaluation, and cross-validation functions required by the examples are provided as standalone R files in this directory. Therefore, the supplied examples can be run without installing the complete `manysvms` package.

Users who wish to reproduce or extend additional model-comparison experiments may install the full package as described below.

## Software requirements

The code was developed and tested using R 4.3.2 under Windows 11. R 4.0 or later is recommended.

The required R packages can be installed using:

```r
required_packages <- c(
  "mvtnorm",
  "ggplot2",
  "foreach",
  "doSNOW",
  "iterators"
)

new_packages <- required_packages[
  !required_packages %in% rownames(installed.packages())
]

if (length(new_packages) > 0) {
  install.packages(new_packages)
}
```

## Quick start

Clone the repository and move to the RaENSVM directory:

```bash
git clone https://github.com/LiFeiH/SVMs-and-SVRs.git
cd "SVMs-and-SVRs/RaENSVM code"
```

Run the complete example using:

```bash
Rscript "Example code.R"
```

Alternatively, open `Example code.R` in RStudio, set `RaENSVM code` as the working directory, and run the complete script.

The script successively demonstrates:

1. direct fitting of a linear RaENSVM on simulated data;
2. hyperparameter selection by five-fold cross-validation;
3. cross-validation under label noise; and
4. fitting RaENSVM with a Gaussian kernel.

The output includes a confusion matrix, a fitted decision-boundary plot, and summaries of the cross-validation results.

## Files

- `RaENSVM.R`: core implementation of RaENSVM.
- `Kernel Function.R`: kernel-construction functions.
- `Cross Validation function.R`: prediction, grid-search, and cross-validation functions.
- `Metric.R`: classification evaluation metrics.
- `gg.R`: plotting settings used in the simulated example.
- `Example code.R`: complete usage examples.
- `Algerian_forest_fires_dataset_UPDATE.csv`: dataset used in the cross-validation examples.
- `The code of other models/`: implementations of comparison methods used in related experiments.

## Optional installation of `manysvms`

The full `manysvms` package is not required for the supplied examples. It can be installed for access to additional SVM implementations and supporting functions:

```r
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

devtools::install_github("define957/manysvms")
```

## Data source

The Algerian Forest Fires dataset was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/547/algerian%2Bforest%2Bfire%2Bdataset).

## Reproducibility

Random seeds and parameter grids are specified in `Example code.R`. The supplied examples illustrate the implementation and use of RaENSVM but are not intended to reproduce every table and figure reported in the manuscript.
