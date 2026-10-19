# Polynomial Regression: Fitting, Generalization & Regularization

Simulation study of how model complexity, sample size, and noise level control generalization (CSI 5340 Deep Learning, Homework 1).

## Background

The core question of learning theory: a model that fits the training data well may fail on new data. This project makes that concrete with a fully controlled simulation — the true function is known (`Y = cos(2πX) + Gaussian noise`, `X ~ Uniform(0,1)`), so in-sample error (**Ein**), out-of-sample error (**Eout**), and **bias** (error of the average hypothesis, **Ebias**) can all be measured exactly. Everything is implemented from scratch in NumPy, including manual gradient descent.

## The Assignment

- Implement `getData` (generate N samples), `getMSE`, `fitData` (fit a degree-d polynomial with **manually-coded** gradient descent — no autograd for the update step), and `experiment` (average Ein/Eout over M≥20 trials, plus Ebias from the averaged hypothesis on a 1000-point test set).
- Vary three factors one at a time, with and without weight-decay regularization: model complexity d ∈ {0…20}, sample size N ∈ {2,5,10,20,50,100,200}, noise σ ∈ {0.01, 0.1, 1}.
- Deliverables: all code plus a report of findings.

## Implementation

- `hw1.py` — everything from scratch in NumPy (no autograd): data generation, MSE, analytic gradients, GD loop (lr=0.01, 1000 epochs, weight decay 0 / 0.1), and the experiment harness.
- Full factorial grid: 7 sample sizes × 21 polynomial degrees × 3 noise levels × 2 regularization settings = 882 configurations, each averaged over 50 trials → **44,100 gradient-descent fits**, every one evaluated on a 1000-point test set for Eout/Ebias. Results → `results.csv` / `results_reg.csv`.
- `hw1plot.py` — Ein/Eout/Ebias curves from the CSVs; output figures in `results/`.

## Key Findings

- **Sample size**: as N grows, Eout drops sharply while Ein rises — they converge. For the complex model (d=17), the generalization gap (Eout−Ein) is huge when N<10: complex models need data to earn their capacity.
- **Model complexity**: Eout bottoms out at d=9 — the sweet spot dividing underfitting from overfitting — then rises while Ein keeps falling. More data (N=100 vs 20) visibly shrinks overfitting.
- **Noise**: as σ goes 0.01→1, Eout/Ebias explode (0.35→2.1/0.9) while Ein barely moves — noise hurts generalization, not fitting. At high noise, only more data (N=200) closes the gap; regularization alone can't.
- **Weight decay ≈ "soft" data/complexity adjustment**: it lowers early Eout/Ebias peaks, makes the generalization gap vanish earlier (like having more data), and shifts the optimal complexity earlier (d=10 vs 11 — like using a simpler model).

## How to Run

```bash
pip install -r requirements.txt
python hw1.py        # runs the full grid (~hours); writes results.csv / results_reg.csv
python hw1plot.py    # reproduces the figures in results/
```

## Repository Structure

```
├── hw1.py            # simulation: data, GD fitting, experiments
├── hw1plot.py        # figure generation
├── results/          # Ein/Eout/Ebias plots (with/without regularization)
└── report/
    └── Assignment1.pdf
```
