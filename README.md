# Federated Learning and Differential Privacy Framework
## Overview
This module implements a full privacy-preserving ML training framework combining federated 
learning (FL), differential privacy (DP), and secure communication. The system simulates 
multiple independent organizations contributing to a shared global model while keeping 
their data local.

## Architecture
- **Local Clients (Hospitals)**: train model on private data
- **Coordinator**: aggregates gradients using FedAvg
- **Privacy Layer**: DP-SGD with noise addition
- **Security Layer**: optional PQC for model updates
- **Dashboard**: training progress monitoring

## Getting Started
1. Install dependencies  
2. Run `python generate_synthetic_data.py`  
3. Launch federated training via `train_federated.py`  
4. Open dashboard using `streamlit run dashboard/app.py`  

## Results
- Model performance comparison: centralized vs federated
- Privacy budget analysis (ε values)
- Communication overhead benchmarks
