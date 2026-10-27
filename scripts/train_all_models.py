"""
Master Model Training Runner
Trains all 5 clinical disease models sequentially and saves metrics to JSON files.
"""

import subprocess
import sys
import os

print("=" * 70)
print("DISEASE PREDICTION SYSTEM - TRAINING ALL MODELS")
print("=" * 70)
print("\nThis script will train models for all 5 diseases using UCI datasets.\n")

script_dir = os.path.dirname(os.path.abspath(__file__))

scripts = [
    ("Breast Cancer", "train_breast_cancer_model.py"),
    ("Diabetes", "train_diabetes_model.py"),
    ("Heart Disease", "train_heart_disease_model.py"),
    ("Kidney Disease", "train_kidney_disease_model.py"),
    ("Liver Disease", "train_liver_disease_model.py"),
]

successful = []
failed = []

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

for disease_name, script_filename in scripts:
    print(f"\n{'=' * 70}")
    print(f"Training {disease_name} Model...")
    print(f"Running: {script_filename}")
    print("=" * 70)
    
    script_full_path = os.path.join(script_dir, script_filename)
    
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", script_full_path],
            cwd=script_dir,
            env=env,
            check=True
        )
        successful.append(disease_name)
        print(f"[OK] {disease_name} model trained successfully!")
    except subprocess.CalledProcessError as e:
        failed.append(disease_name)
        print(f"[ERROR] Error training {disease_name} model: {e}")
    except Exception as e:
        failed.append(disease_name)
        print(f"[ERROR] Unexpected error for {disease_name}: {e}")

print(f"\n{'=' * 70}")
print("TRAINING SUMMARY")
print("=" * 70)
print(f"\nSuccessfully trained: {len(successful)}/5")
for disease in successful:
    print(f"  [OK] {disease}")

if failed:
    print(f"\nFailed to train: {len(failed)}/5")
    for disease in failed:
        print(f"  [FAIL] {disease}")
else:
    print(f"\n[SUCCESS] All 5 disease models trained successfully!")

print("\nAll model metrics have been saved to respective JSON files in scripts/.")
print("=" * 70)
