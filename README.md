# Federated Learning Research

This repository serves as a workspace for implementing, studying, and experimenting with various Federated Learning (FL) algorithms. The goal is to build a collection of reference implementations and research experiments, growing from fundamental algorithms to advanced personalized techniques.

## Current Implementations

### pFedMe (Personalized Federated Learning with Moreau Envelopes)
**File:** `pfedme.py`

An implementation of the **pFedMe** algorithm, which tackles the statistical heterogeneity in FL by optimizing a personalized model for each client using Moreau Envelopes. This bi-level optimization approach allows clients to pursue personalized models while still contributing to a robust global model.

**Key Features Implemented:**
- **Client Class:** Handles local personalized optimization (inner loop) using gradient descent on the Moreau envelope.
- **Server Class:** Manages the global model aggregation and broadcasting.
- **Bi-level Optimization:** Implements the specific $\theta$ (personalized) and $w$ (global) update rules.

**Scenarios:**
The script includes two synthetic regression scenarios to demonstrate convergence behavior under different conditions:
1.  **High Convexity:** A well-conditioned problem (Condition Number $\approx$ 1). Convergence is fast and stable.
2.  **Low Convexity:** An ill-conditioned problem (Condition Number $\approx$ 100). Convergence is slower, testing the algorithm's robustness.

## Getting Started

### Prerequisites
*   Python 3.x
*   `numpy`
*   `matplotlib`

Install the required packages:
```bash
pip install numpy matplotlib
```

### Usage
Run the main script to execute the pFedMe simulation:

```bash
python pfedme.py
```

### Outputs
The script will print the average personalized loss for each scenario during training and generate a convergence plot named `pfedme_results.png` in the current directory.

## Future Plans
- [ ] Implement FedAvg (Federated Averaging) as a baseline.
- [ ] Add more complex non-convex datasets (e.g., MNIST/CIFAR).
- [ ] Explore other personalized FL algorithms (e.g., Per-FedAvg, Ditto).
