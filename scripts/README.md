# Model Training Guide

This directory contains machine learning training scripts for 5 clinical conditions:

1. **Diabetes**: Pima Indians Diabetes Dataset
2. **Heart Disease**: Cleveland Heart Disease Dataset
3. **Breast Cancer**: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
4. **Kidney Disease**: Chronic Kidney Disease (CKD) Dataset
5. **Liver Disease**: Indian Liver Patient Dataset (ILPD)

## Prerequisites

Install Python dependencies:
```bash
pip install numpy pandas scikit-learn
```

## Training

### Train all models at once:
```bash
python train_all_models.py
```

### Train individually:
```bash
python train_diabetes_model.py
python train_heart_disease_model.py
python train_breast_cancer_model.py
python train_kidney_disease_model.py
python train_liver_disease_model.py
```

## Output

Each training run produces performance metrics (`accuracy`, `precision`, `recall`, `f1_score`, `auc_roc`, and `top_features`) exported as JSON files:
- `diabetes_metrics.json`
- `heart_disease_metrics.json`
- `breast_cancer_metrics.json`
- `kidney_disease_metrics.json`
- `liver_disease_metrics.json`
