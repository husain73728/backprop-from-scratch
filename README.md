# 🧠 Simple Neural Network from Scratch (Pure Python)

## 📌 Overview
This project implements a **fully connected Artificial Neural Network (ANN)** from scratch using only **pure Python (no NumPy, no libraries)**.

The goal is to understand the core mechanics of neural networks, including:
- Forward propagation
- Backpropagation
- Gradient descent
- Activation functions

All computations are performed manually using loops, making it ideal for learning and debugging neural network internals.

---

## 🧱 Model Architecture
**3 → 6 → 4 → 1**

| Layer            | Neurons | Activation |
|------------------|--------|------------|
| Input Layer      | 3      | —          |
| Hidden Layer 1   | 6      | Sigmoid    |
| Hidden Layer 2   | 4      | Sigmoid    |
| Output Layer     | 1      | Sigmoid    |

---

## ⚙️ Features
- Neural network built completely from scratch
- No external libraries (except `math`)
- Manual implementation of:
  - Weight initialization
  - Bias handling
  - Forward propagation
  - Backpropagation
  - Gradient descent updates
- Mean Squared Error (MSE) loss

---

## 📊 Dataset
A small synthetic dataset is used for demonstration:

- Input: 3 features
- Output: Binary (0 or 1)

---

## 🔁 Training Process
- Loss Function: **Mean Squared Error**
- Activation Function: **Sigmoid**
- Optimization: **Gradient Descent**
- Epochs: `15000`
- Learning Rate: `0.1`

---

## ▶️ How to Run

```bash
python ann.py
