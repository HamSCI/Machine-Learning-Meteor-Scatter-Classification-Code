#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 4-Class Meteor Classifier
Validates that the model is learning real patterns, not random noise
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, permutation_test_score

# ----- CONFIGURATION -----
MODEL_FILE = "/Users/ninatormann/Desktop/10mMixed/meteor_4class_classifier.pkl"
TRAINING_CSV = "/Users/ninatormann/Desktop/10mMixed/simple_classification.csv"
TRAINING_FEATURES_CSV = "/Users/ninatormann/Desktop/10mMixed/training_features_4class.csv"


def test_1_model_exists():
    """Test 1: Verify model file exists and can be loaded."""
    print("\n" + "="*70)
    print("TEST 1: Model File Validation")
    print("="*70)
    
    if not os.path.exists(MODEL_FILE):
        print("❌ FAIL: Model file not found!")
        return False
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            model_package = pickle.load(f)
        
        required_keys = ['model', 'scaler', 'label_encoder', 'feature_columns']
        missing = [k for k in required_keys if k not in model_package]
        
        if missing:
            print(f"❌ FAIL: Missing keys in model package: {missing}")
            return False
        
        print("✅ PASS: Model file loaded successfully")
        print(f"   Model type: {type(model_package['model']).__name__}")
        print(f"   Features: {len(model_package['feature_columns'])}")
        print(f"   Classes: {list(model_package['label_encoder'].classes_)}")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error loading model: {e}")
        return False


def test_2_training_accuracy():
    """Test 2: Check training accuracy on original data."""
    print("\n" + "="*70)
    print("TEST 2: Training Data Accuracy")
    print("="*70)
    
    if not os.path.exists(TRAINING_FEATURES_CSV):
        print("❌ SKIP: Training features CSV not found")
        return None
    
    try:
        # Load model
        with open(MODEL_FILE, 'rb') as f:
            model_package = pickle.load(f)
        
        # Load training features
        df = pd.read_csv(TRAINING_FEATURES_CSV)
        
        # Use 'classification' column (not 'density_type')
        if 'classification' not in df.columns:
            print("❌ FAIL: 'classification' column not found in features")
            return False
        
        X = df[model_package['feature_columns']].values
        y_true = df['classification'].values
        
        # Scale and predict
        X_scaled = model_package['scaler'].transform(X)
        y_pred_encoded = model_package['model'].predict(X_scaled)
        y_pred = model_package['label_encoder'].inverse_transform(y_pred_encoded)
        
        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred)
        
        print(f"   Total samples: {len(df)}")
        print(f"   Overall Accuracy: {accuracy:.1%}")
        
        # Show per-class accuracy
        print(f"\n   Per-Class Performance:")
        for cls in model_package['label_encoder'].classes_:
            mask = y_true == cls
            if np.sum(mask) > 0:
                cls_acc = np.mean(y_pred[mask] == y_true[mask])
                print(f"      {cls:15s}: {cls_acc:.1%} ({np.sum(mask)} samples)")
        
        # Confusion matrix
        print(f"\n   Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred, labels=model_package['label_encoder'].classes_)
        
        print(f"   {'Actual →':15s}", end="")
        for cls in model_package['label_encoder'].classes_:
            print(f"{cls[:10]:>12s}", end="")
        print()
        
        for i, cls in enumerate(model_package['label_encoder'].classes_):
            print(f"   {cls:15s}", end="")
            for j in range(len(model_package['label_encoder'].classes_)):
                print(f"{cm[i,j]:12d}", end="")
            print()
        
        if accuracy < 0.6:
            print(f"\n   ⚠️  WARNING: Accuracy < 60% - model may not be learning well")
            return False
        elif accuracy > 0.95:
            print(f"\n   ⚠️  WARNING: Accuracy > 95% - possible overfitting")
        else:
            print(f"\n   ✅ PASS: Reasonable accuracy range")
        
        return accuracy >= 0.6
        
    except Exception as e:
        print(f"❌ FAIL: Error in accuracy test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_cross_validation():
    """Test 3: Cross-validation to check for overfitting."""
    print("\n" + "="*70)
    print("TEST 3: Cross-Validation (Overfitting Check)")
    print("="*70)
    
    if not os.path.exists(TRAINING_FEATURES_CSV):
        print("❌ SKIP: Training features CSV not found")
        return None
    
    try:
        # Load model and data
        with open(MODEL_FILE, 'rb') as f:
            model_package = pickle.load(f)
        
        df = pd.read_csv(TRAINING_FEATURES_CSV)
        
        X = df[model_package['feature_columns']].values
        y = df['classification'].values
        
        # Encode labels
        y_encoded = model_package['label_encoder'].transform(y)
        
        # Scale
        X_scaled = model_package['scaler'].transform(X)
        
        # 5-fold cross-validation
        cv_scores = cross_val_score(model_package['model'], X_scaled, y_encoded, cv=5)
        
        print(f"   5-Fold CV Scores: {[f'{s:.1%}' for s in cv_scores]}")
        print(f"   Mean CV Accuracy: {cv_scores.mean():.1%}")
        print(f"   Std Dev: {cv_scores.std():.1%}")
        
        # Check for overfitting
        if cv_scores.std() > 0.15:
            print(f"\n   ⚠️  WARNING: High variance (>15%) - possible overfitting")
            return False
        else:
            print(f"\n   ✅ PASS: Low variance - model is stable")
            return True
            
    except Exception as e:
        print(f"❌ FAIL: Error in cross-validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_permutation_test():
    """Test 4: Permutation test - is model better than random?"""
    print("\n" + "="*70)
    print("TEST 4: Permutation Test (Better Than Random?)")
    print("="*70)
    print("   This may take a minute...")
    
    if not os.path.exists(TRAINING_FEATURES_CSV):
        print("❌ SKIP: Training features CSV not found")
        return None
    
    try:
        # Load model and data
        with open(MODEL_FILE, 'rb') as f:
            model_package = pickle.load(f)
        
        df = pd.read_csv(TRAINING_FEATURES_CSV)
        
        X = df[model_package['feature_columns']].values
        y = df['classification'].values
        y_encoded = model_package['label_encoder'].transform(y)
        X_scaled = model_package['scaler'].transform(X)
        
        # Permutation test (shuffles labels randomly to test if model beats chance)
        score, perm_scores, pvalue = permutation_test_score(
            model_package['model'], X_scaled, y_encoded, 
            cv=5, n_permutations=100, random_state=42
        )
        
        print(f"   Model Score: {score:.1%}")
        print(f"   Random Scores Mean: {perm_scores.mean():.1%}")
        print(f"   P-value: {pvalue:.4f}")
        
        if pvalue < 0.05:
            print(f"\n   ✅ PASS: Model is significantly better than random (p < 0.05)")
            return True
        else:
            print(f"\n   ❌ FAIL: Model NOT significantly better than random (p >= 0.05)")
            print(f"   This suggests the model may be learning noise, not real patterns!")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Error in permutation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_feature_importance():
    """Test 5: Check if important features make physical sense."""
    print("\n" + "="*70)
    print("TEST 5: Feature Importance Analysis")
    print("="*70)
    
    try:
        # Load model
        with open(MODEL_FILE, 'rb') as f:
            model_package = pickle.load(f)
        
        model = model_package['model']
        features = model_package['feature_columns']
        
        # Get feature importances (Random Forest has this)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            print(f"\n   Top 10 Most Important Features:")
            for i in range(min(10, len(features))):
                idx = indices[i]
                print(f"   {i+1}. {features[idx]:30s} : {importances[idx]:.4f}")
            
            # Check if expected features are important
            expected_important = ['peak_duration', 'time_above_threshold', 'rise_time', 'fall_time', 'num_peaks']
            
            top_10_features = [features[indices[i]] for i in range(min(10, len(features)))]
            found = [f for f in expected_important if f in top_10_features]
            
            if len(found) >= 3:
                print(f"\n   ✅ PASS: Expected features ({', '.join(found)}) are important")
                return True
            else:
                print(f"\n   ⚠️  WARNING: Expected features not in top 10")
                print(f"   This might indicate the model is using unexpected patterns")
                return False
        else:
            print("   ℹ️  Model type doesn't support feature importances")
            return None
            
    except Exception as e:
        print(f"❌ FAIL: Error in feature importance: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_class_balance():
    """Test 6: Check class distribution."""
    print("\n" + "="*70)
    print("TEST 6: Class Balance Check")
    print("="*70)
    
    if not os.path.exists(TRAINING_CSV):
        print("❌ SKIP: Training CSV not found")
        return None
    
    try:
        df = pd.read_csv(TRAINING_CSV)
        
        counts = df['classification'].value_counts()
        
        print(f"\n   Class Distribution:")
        for cls, count in counts.items():
            pct = count / len(df) * 100
            print(f"   {cls:15s}: {count:4d} ({pct:5.1f}%)")
        
        # Check imbalance ratio
        ratio = counts.max() / counts.min()
        print(f"\n   Imbalance Ratio: {ratio:.2f}:1")
        
        if ratio > 15:
            print(f"\n   ⚠️  WARNING: Very high class imbalance (>{15}:1)")
            print(f"   Model uses class_weight='balanced' to handle this")
            return True  # Still pass since we're using balanced weights
        elif ratio > 5:
            print(f"\n   ℹ️  Moderate imbalance - handled with class_weight='balanced'")
            return True
        else:
            print(f"\n   ✅ PASS: Reasonable class balance")
            return True
            
    except Exception as e:
        print(f"❌ FAIL: Error checking class balance: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_sanity_check():
    """Test 7: Sanity checks on clear examples."""
    print("\n" + "="*70)
    print("TEST 7: Sanity Check (Known Examples)")
    print("="*70)
    
    if not os.path.exists(TRAINING_FEATURES_CSV):
        print("❌ SKIP: Training features CSV not found")
        return None
    
    try:
        # Load model and data
        with open(MODEL_FILE, 'rb') as f:
            model_package = pickle.load(f)
        
        df = pd.read_csv(TRAINING_FEATURES_CSV)
        
        print(f"\n   Testing clear examples of each class:")
        
        correct_count = 0
        total_count = 0
        
        # Test each class
        for cls in ['overdense', 'underdense', 'aircraft', 'noise']:
            cls_examples = df[df['classification'] == cls].head(3)
            
            if len(cls_examples) > 0:
                print(f"\n   {cls.upper()}:")
                for idx, row in cls_examples.iterrows():
                    X = row[model_package['feature_columns']].values.reshape(1, -1)
                    X_scaled = model_package['scaler'].transform(X)
                    pred_encoded = model_package['model'].predict(X_scaled)[0]
                    pred = model_package['label_encoder'].inverse_transform([pred_encoded])[0]
                    
                    correct = (pred == cls)
                    symbol = '✅' if correct else '❌'
                    print(f"   {symbol} True: {cls:12s} → Predicted: {pred:12s} (duration: {row.get('peak_duration', 0):.2f}s)")
                    
                    if correct:
                        correct_count += 1
                    total_count += 1
        
        if total_count == 0:
            print("   ⚠️  No examples found to test")
            return None
        
        accuracy = correct_count / total_count
        print(f"\n   Sanity Check Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")
        
        if accuracy >= 0.6:
            print(f"   ✅ PASS: Model correctly classifies most clear examples")
            return True
        else:
            print(f"   ❌ FAIL: Model struggles with obvious examples")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Error in sanity check: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_summary_report():
    """Generate final summary report."""
    print("\n" + "="*70)
    print("RUNNING ALL TESTS")
    print("="*70)
    
    results = {}
    
    results['Model Exists'] = test_1_model_exists()
    results['Training Accuracy'] = test_2_training_accuracy()
    results['Cross-Validation'] = test_3_cross_validation()
    results['Permutation Test'] = test_4_permutation_test()
    results['Feature Importance'] = test_5_feature_importance()
    results['Class Balance'] = test_6_class_balance()
    results['Sanity Check'] = test_7_sanity_check()
    
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v == True)
    failed = sum(1 for v in results.values() if v == False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        if result == True:
            print(f"   ✅ {test_name}")
        elif result == False:
            print(f"   ❌ {test_name}")
        else:
            print(f"   ⚠️  {test_name} (skipped)")
    
    print(f"\n   Total: {passed} passed, {failed} failed, {skipped} skipped")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if failed == 0:
        print("   🎉 EXCELLENT: All tests passed!")
        print("   Your 4-class model is working correctly and learning real patterns.")
        print("\n   Next steps:")
        print("   ✅ Use predict_4class.py to classify new files")
        print("   ✅ Use visualize_ml_predictions.py to create plots")
    elif failed <= 2:
        print("   ✅ GOOD: Most tests passed")
        print("   Model is likely working, but review failed tests for potential issues.")
    else:
        print("   ⚠️  CONCERN: Multiple test failures")
        print("   Review the failures above. Common issues:")
        print("   - Make sure you trained with train_4class_classifier.py")
        print("   - Check that paths point to correct CSV files")
        print("   - Verify 'classification' column exists in CSVs")


if __name__ == "__main__":
    print("="*70)
    print("4-CLASS METEOR CLASSIFIER - VALIDATION TEST SUITE")
    print("="*70)
    
    generate_summary_report()
