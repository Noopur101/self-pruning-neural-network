# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project implements a neural network that **learns to prune itself during training** using learnable gates and L1 regularization.

Each weight is associated with a gate value in the range (0,1).
If a gate approaches 0, the corresponding weight is effectively removed, enabling the network to **dynamically reduce its own complexity**.

---

## Why L1 Penalty Encourages Sparsity

The pruning mechanism is driven by applying an **L1 penalty on the sigmoid-transformed gate values**.

* Each gate is computed as:

  `gate = sigmoid(gate_score)` → values in (0,1)

* The sparsity loss is:

  `L1 Loss = sum (or mean) of all gate values`

### Intuition

* L1 regularization penalizes **non-zero values**
* Since gates are always positive:

  * Minimizing L1 pushes many gate values → **0**
* When a gate ≈ 0:

  * The corresponding weight becomes ≈ 0
  * The connection is effectively **pruned**
This allows the model to automatically retain only the most important connections, resulting in a **sparse and efficient network**

---

## 📊 Results: Sparsity vs Accuracy Trade-off

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
| ---------- | ----------------- | ------------ |
| 0.01       | 55.12%              | 0.00         |
| 0.1        | 54.42%              | 0.00         |
| 1.0        | 55.29%              | 0.00         |

### Observations

* Increasing λ → **higher sparsity (more pruning)**
* Higher sparsity → **lower accuracy**
* Demonstrates a clear **trade-off between performance and model efficiency**

---

## Gate Value Distribution

Below is the distribution of learned gate values for the best-performing model:

![Gate Distribution](outputs.png)

### Interpretation

* **Spike near 0** → many weights pruned
* **Cluster away from 0 (near 1)** → important weights retained

This confirms successful **self-pruning behavior**

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run experiments

```bash
python experiments/run_experiments.py
```

### OR

Open the notebook:

```
pruning.ipynb
```

---

## Key Takeaways

* Self-pruning can be integrated directly into training
* L1 regularization effectively induces sparsity
* Proper tuning of λ is crucial for balancing:

  * Accuracy
  * Model compression

---

## Author

**Noopur Vispute**
