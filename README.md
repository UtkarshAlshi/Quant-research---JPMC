# JPMC Quantitative Research Certification - Project Tasks

## Overview
This repository contains solutions to various tasks completed as part of the **JPMC Quantitative Research Certification**. These tasks focus on building predictive models, financial analysis, and quantization techniques, which are critical for risk management and trading desk operations. The solutions include Python code for pricing models, probability of default (PD) estimation, and FICO score quantization.

---

## Task Descriptions

### 1. **Natural Gas Storage Contract Pricing Model**
   **Objective**: Develop a pricing model for natural gas storage contracts. The model calculates the value of a contract based on gas purchase prices, storage costs, and expected future prices.

   **Key Concepts**:
   - Buy and sell prices of natural gas over a period.
   - Fixed storage costs, injection/withdrawal fees, and transport costs.
   - Overall contract valuation formula:
     \[
     \text{Value} = (\text{Sell Price} - \text{Buy Price}) \times \text{Quantity} - \text{Storage Costs} - \text{Injection/Withdrawal Fees} - \text{Transport Fees}
     \]

   **Tech Stack**: Python (Pandas, NumPy)

### 2. **Probability of Default (PD) Prediction Model**
   **Objective**: Build a machine learning model that predicts the probability of a borrower defaulting on their loan based on borrower characteristics like income, total loans outstanding, and previous defaults.

   **Key Concepts**:
   - Train a logistic regression model to estimate PD.
   - Use customer metrics such as income and outstanding loans to make predictions.
   - Calculate expected losses using a recovery rate of 10%.

   **Tech Stack**: Python (Scikit-learn, Pandas, NumPy)

### 3. **Quantization of FICO Scores for Mortgage Default Prediction**
   **Objective**: Develop a quantization strategy to map FICO scores into buckets, simplifying the input for models that predict the probability of default on mortgages.

   **Key Concepts**:
   - FICO score ranges from 300 to 850, which need to be mapped into a few buckets.
   - Explore multiple quantization techniques, such as minimizing mean squared error (MSE) and maximizing log-likelihood.
   - Optimize bucket boundaries dynamically to maximize the accuracy of predictions.

   **Tech Stack**: Python (Dynamic Programming, NumPy, Pandas)

---

## Installation
1. Clone the repository.
   ```bash
   git clone <repository-url>
   ```
2. Install the necessary dependencies.
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run
- Each task is in its own Python file for easy execution. Run any task using Python:
   ```bash
   python task_<number>.py
   ```
   - Example:
     ```bash
     python task_1.py
     ```

---

## Results
Each task outputs the relevant results based on the provided inputs, such as:
- Predicted value of a natural gas storage contract.
- Predicted probability of default and expected losses for personal loans.
- Optimal FICO score buckets for mortgage default prediction.

---

## Contributors
- **Utkarsh Alshi**  
   JPMC Quantitative Research Certification Participant

