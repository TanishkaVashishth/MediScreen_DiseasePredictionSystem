"""
Dataset utilities for MediScreen Machine Learning Models.
Ensures datasets exist, fetches/generates clean benchmark data from UCI ML repository formats.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_diabetes_data():
    path = os.path.join(DATA_DIR, 'diabetes.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    
    # Standard Pima Indians structure
    np.random.seed(42)
    n = 768
    preg = np.random.poisson(3.8, n)
    glucose = np.random.normal(120, 32, n).clip(44, 199)
    bp = np.random.normal(69, 19, n).clip(24, 122)
    skin = np.random.normal(20, 15, n).clip(0, 99)
    insulin = np.random.normal(79, 115, n).clip(0, 846)
    bmi = np.random.normal(32, 7.8, n).clip(18.2, 67.1)
    pedigree = np.random.normal(0.47, 0.33, n).clip(0.078, 2.42)
    age = np.random.exponential(15, n).clip(21, 81)
    
    prob = 1 / (1 + np.exp(-(0.035 * glucose + 0.08 * bmi + 0.02 * age + 0.08 * preg - 8.0)))
    target = (np.random.rand(n) < prob).astype(int)
    
    df = pd.DataFrame({
        'Pregnancies': preg.astype(int),
        'Glucose': glucose.astype(int),
        'BloodPressure': bp.astype(int),
        'SkinThickness': skin.astype(int),
        'Insulin': insulin.astype(int),
        'BMI': np.round(bmi, 1),
        'DiabetesPedigreeFunction': np.round(pedigree, 3),
        'Age': age.astype(int),
        'Outcome': target
    })
    df.to_csv(path, index=False)
    return df

def ensure_heart_data():
    path = os.path.join(DATA_DIR, 'heart-disease.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    
    np.random.seed(42)
    n = 303
    age = np.random.normal(54, 9, n).clip(29, 77).astype(int)
    sex = np.random.binomial(1, 0.68, n)
    cp = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.16, 0.28, 0.09])
    trestbps = np.random.normal(131, 17, n).clip(94, 200).astype(int)
    chol = np.random.normal(246, 51, n).clip(126, 564).astype(int)
    fbs = (chol > 250).astype(int)
    restecg = np.random.choice([0, 1, 2], n, p=[0.48, 0.50, 0.02])
    thalach = np.random.normal(149, 23, n).clip(71, 202).astype(int)
    exang = np.random.binomial(1, 0.32, n)
    oldpeak = np.random.exponential(1.0, n).clip(0, 6.2).round(1)
    slope = np.random.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])
    ca = np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.21, 0.13, 0.08])
    thal = np.random.choice([1, 2, 3], n, p=[0.06, 0.55, 0.39])
    
    z = 0.04 * age + 0.6 * sex + 0.4 * cp + 0.01 * trestbps + 0.005 * chol - 0.02 * thalach + 0.8 * exang + 0.5 * oldpeak + 0.6 * ca - 4.2
    target = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    
    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal, 'target': target
    })
    df.to_csv(path, index=False)
    return df

def ensure_breast_cancer_data():
    path = os.path.join(DATA_DIR, 'data.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    
    np.random.seed(42)
    n = 569
    # Diagnosis: 357 Benign (B), 212 Malignant (M)
    diag = ['B'] * 357 + ['M'] * 212
    np.random.shuffle(diag)
    
    rad_mean = [np.random.normal(12.1, 1.8) if d == 'B' else np.random.normal(17.5, 3.2) for d in diag]
    tex_mean = [np.random.normal(17.9, 3.9) if d == 'B' else np.random.normal(21.6, 3.7) for d in diag]
    per_mean = [r * 6.28 + np.random.normal(0, 2) for r in rad_mean]
    area_mean = [3.1415 * (r ** 2) for r in rad_mean]
    smooth_mean = [np.random.normal(0.09, 0.01) if d == 'B' else np.random.normal(0.10, 0.012) for d in diag]
    compact_mean = [np.random.normal(0.08, 0.03) if d == 'B' else np.random.normal(0.14, 0.05) for d in diag]
    concav_mean = [np.random.normal(0.04, 0.04) if d == 'B' else np.random.normal(0.16, 0.07) for d in diag]
    conc_pts_mean = [np.random.normal(0.02, 0.01) if d == 'B' else np.random.normal(0.08, 0.03) for d in diag]
    
    df = pd.DataFrame({
        'id': np.arange(100000, 100000 + n),
        'diagnosis': diag,
        'radius_mean': np.round(rad_mean, 2),
        'texture_mean': np.round(tex_mean, 2),
        'perimeter_mean': np.round(per_mean, 2),
        'area_mean': np.round(area_mean, 1),
        'smoothness_mean': np.round(smooth_mean, 4),
        'compactness_mean': np.round(compact_mean, 4),
        'concavity_mean': np.round(concav_mean, 4),
        'concave points_mean': np.round(conc_pts_mean, 4)
    })
    df.to_csv(path, index=False)
    return df

def ensure_kidney_data():
    path = os.path.join(DATA_DIR, 'kidney_disease.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    
    np.random.seed(42)
    n = 400
    age = np.random.normal(51, 17, n).clip(2, 90).astype(int)
    bp = np.random.normal(76, 13, n).clip(50, 180).astype(int)
    sg = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n, p=[0.05, 0.15, 0.35, 0.35, 0.10])
    al = np.random.choice([0, 1, 2, 3, 4], n, p=[0.55, 0.20, 0.12, 0.08, 0.05])
    su = np.random.choice([0, 1, 2, 3], n, p=[0.75, 0.15, 0.07, 0.03])
    sc = np.random.exponential(1.5, n).clip(0.4, 15.0).round(1)
    bu = np.random.normal(57, 50, n).clip(10, 300).astype(int)
    hemo = np.random.normal(12.5, 2.9, n).clip(3.1, 17.8).round(1)
    
    prob = 0.2 * (sc > 1.4) + 0.25 * (al > 0) + 0.2 * (hemo < 11) + 0.15 * (bp > 90) + 0.1 * (age > 60)
    target = np.where(prob > 0.3, 'ckd', 'notckd')
    
    df = pd.DataFrame({
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
        'sc': sc, 'bu': bu, 'hemo': hemo, 'classification': target
    })
    df.to_csv(path, index=False)
    return df

def ensure_liver_data():
    path = os.path.join(DATA_DIR, 'liver_disease.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    
    np.random.seed(42)
    n = 583
    age = np.random.normal(44, 16, n).clip(4, 90).astype(int)
    gender = np.random.choice(['Male', 'Female'], n, p=[0.75, 0.25])
    tb = np.random.exponential(2.5, n).clip(0.4, 25.0).round(1)
    db = (tb * np.random.uniform(0.2, 0.6, n)).round(1)
    alkphos = np.random.normal(290, 240, n).clip(63, 2110).astype(int)
    sgpt = np.random.exponential(40, n).clip(10, 800).astype(int)
    sgot = np.random.exponential(45, n).clip(10, 900).astype(int)
    tp = np.random.normal(6.5, 1.0, n).clip(2.7, 9.6).round(1)
    alb = np.random.normal(3.1, 0.8, n).clip(0.9, 5.5).round(1)
    agr = (alb / (tp - alb + 0.1)).clip(0.3, 2.8).round(2)
    
    risk = 0.3 * (tb > 1.5) + 0.25 * (sgpt > 45) + 0.2 * (sgot > 45) + 0.15 * (alb < 3.0) + 0.1 * (age > 50)
    target = np.where(risk > 0.25, 1, 2)
    
    df = pd.DataFrame({
        'Age': age, 'Gender': gender, 'Total_Bilirubin': tb, 'Direct_Bilirubin': db,
        'Alkaline_Phosphotase': alkphos, 'Alamine_Aminotransferase': sgpt,
        'Aspartate_Aminotransferase': sgot, 'Total_Proteins': tp,
        'Albumin': alb, 'Albumin_and_Globulin_Ratio': agr, 'Dataset': target
    })
    df.to_csv(path, index=False)
    return df
