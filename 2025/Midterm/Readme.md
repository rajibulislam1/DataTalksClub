# Nursery School Application Classification

## Overview
This project predicts the suitability of a child’s application for a nursery school using the Nursery dataset. The dataset contains categorical information about family structure, parents' employment, social and health conditions, and more.

## Problem Statement
The goal is to build a classification model that can accurately predict the final evaluation of a nursery application (`NURSERY`) based on 8 input attributes:
- `parents` – Parents’ occupation
- `has_nurs` – Child's nursery status
- `form` – Form of the family
- `children` – Number of children
- `housing` – Housing conditions
- `finance` – Financial standing
- `social` – Social conditions
- `health` – Health conditions

## Dataset
- **Instances:** 12,960  
- **Features:** 8 categorical input features  
- **Target:** NURSERY (decision of nursery application)  
- **Source:** [UCI Nursery Dataset](http://www-ai.ijs.si/BlazZupan/nursery.html)

## Methodology
1. Load and preprocess categorical data.
2. Encode categorical features (e.g., using One-Hot or Label Encoding).
3. Train a classification model (e.g., Decision Tree, Random Forest, or XGBoost).
4. Evaluate model performance using accuracy, F1-score, or other metrics.
5. Analyze feature importance and model predictions.

