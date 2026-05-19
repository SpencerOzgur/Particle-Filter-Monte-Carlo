# Particle Filter for Output Gap Estimation

## Overview

This project implements a **Particle Filter (Sequential Monte Carlo)** to estimate a latent economic variable — the **output gap** — from noisy GDP observations.

The output gap is modeled as a hidden state in a **state-space model**, where:
- The latent state evolves over time
- Observations are noisy measurements of that state

The goal is to:
1. Simulate synthetic macroeconomic data
2. Estimate the latent state using a Particle Filter
3. Compare performance against:
   - Kalman Filter (KF)
   - Maximum Likelihood Estimation (MLE)
4. Evaluate estimation accuracy using RMSE and visualization

---

## Model

We consider the following linear Gaussian state-space model:

### State Equation
$$
\[
x_t = \theta x_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)
\]
$$

### Observation Equation
$$
\[
y_t = \phi x_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \tau^2)
\]
$$

Where:
- $$\(x_t\)$$: latent output gap (unobserved)
- $$\(y_t\)$$: observed GDP growth
- $$\(\theta\)$$: persistence parameter
- $$\(\phi\)$$: loading coefficient
- $$\(\sigma^2, \tau^2\)$$: noise variances

---

## Methods

### 1. Particle Filter (Primary Method)
- Sequential Monte Carlo approximation of posterior $$\( p(x_t | y_{1:t}) \)$$
- Steps:
  - Initialization
  - Prediction
  - Weight update
  - Normalization
  - Resampling

### 2. Kalman Filter (Benchmark)
- Closed-form solution for linear Gaussian models
- Provides optimal baseline for comparison

### 3. Maximum Likelihood Estimation (MLE)
- Parameter estimation and/or likelihood-based benchmark

---

## Evaluation

Performance is assessed using:

- **Root Mean Squared Error (RMSE)** between true and estimated states
- Visual comparison of:
  - True latent state vs estimates
  - PF vs KF vs MLE
- Optional statistical comparisons

---

## Repository Structure
