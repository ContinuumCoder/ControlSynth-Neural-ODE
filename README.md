# ControlSynth Neural ODEs (CSODEs)

## Overview

ControlSynth Neural ODEs (CSODEs) represent an advanced class of Neural ODEs designed for modeling complex physical dynamics with high scalability and flexibility. These models are particularly adept at understanding and predicting the behavior of systems described by partial differential equations, benefiting from an additional control term that captures dynamics at various scales. This project explores CSODEs' ability to guarantee convergence despite their nonlinear nature and demonstrates their superiority in learning and prediction compared to traditional Neural ODEs and other variants.

## Installation

### Prerequisites

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Plotly
- sklearn
- skimage
- thop
- fastdtw

### Setup

Clone this repository to your local machine:

```bash
git clone https://github.com/ContinuumCoder/ControlSynth-Neural-ODE
cd ControlSynth-Neural-ODE
```

Install the required Python packages:

```bash
pip install numpy scipy matplotlib plotly scikit-learn scikit-image torch thop fastdtw
```

## Quick Start

To get started with the CSODEs experiments, follow these steps:

1. **Clone the Repository**

   Clone the CSODE repository to your local machine:

   ```bash
   git clone https://github.com/ContinuumCoder/ControlSynth-Neural-ODE
   cd ControlSynth-Neural-ODE
   ```

2. **Install Dependencies**

   Install all required Python packages:

   ```bash
   pip install numpy scipy matplotlib plotly scikit-learn scikit-image torch thop fastdtw
   ```

3. **Run a Preliminary Experiment Example**

   As an example, navigate to the directory containing the preliminary experiment script and execute:

   ```bash
   cd ControlSynth_Neural_ODE/preliminary_experiment_example
   python spirals_experiment.py
   ```

   This script will run the experiment using the ControlSynth Latent ODE model and output the results. It serves as an example to demonstrate how to execute the experiments in this repository.

## Features

- **Simulation Data Generation:** The repository includes algorithms for generating simulation data that mimic complex dynamic systems, essential for testing and validating the models developed.
- **Model Structures:** Showcases detailed model architectures used in experiments, providing insights into the construction and functionality of various Neural ODEs, including CSODEs.


## Research Highlights

- **ControlSynth Neural ODEs:** Introduces a novel structure with an extra control term for enhanced flexibility and scalability.
- **Convergence Guarantees:** Demonstrates how convergence can be assured through linear inequalities, despite the inherent nonlinear properties of the models.
- **Comparative Studies:** Evaluates CSODEs against traditional NODEs, Augmented Neural ODEs (ANODE), and Second Order Neural ODEs (SONODE), showing improved performance in time-series prediction of physical dynamics.
