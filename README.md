# Implementing-Federated-Learning-Algorithms-for-Non-IID-Data-FedAvg-and-FedDF-


## Project Overview

Implementation of Federated Learning algorithms, FedAvg and FedDF, on CIFAR10 dataset with non-IID data simulation using Dirichlet distribution. Additionally, CIFAR100 is used as a reference dataset for ensemble distillation at the server in FedDF.

## Key Features

- Implementation of two popular Federated Learning algorithms:
    - FedAvg (McMahan et al., 2017)
    - FedDF (Lin et al., NeurIPS 2020)
- Non-IID data simulation using Dirichlet distribution
- CIFAR10 dataset for federated learning
- CIFAR100 dataset as reference for ensemble distillation in FedDF
- Custom implementation of ensemble distillation at the server

## Implementation Details

- Framework: TensorFlow 2.x
- Model choice: ResNet8
- Dataset: CIFAR10 (60,000 32x32 color images)
- Non-IID simulation: Dirichlet distribution with varying concentration parameters
- FedAvg algorithm: federated averaging with infrequent communication
- FedDF algorithm: model fusion at the server using ensemble distillation over a reference dataset
- Optimization: Adam optimizer with weight decay

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization)

## Usage

1. Clone repository
2. Install requirements
3. Download CIFAR10 and CIFAR100 datasets
4. Run training script (python (link unavailable))
5. Evaluate model performance on test dataset

## Example Use Cases

- Federated learning for edge devices
- Non-IID data handling in distributed learning
- Ensemble distillation for improved performance

## References

- McMahan et al. (ICAIS 2017) - Communication-Efficient Learning of Deep Networks from Decentralized Data
- Lin et al. (NeurIPS 2020) - Ensemble Distillation for Robust Model Fusion in Federated Learning
