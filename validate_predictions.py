#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate 4-Class Predictions
Helps you spot-check if predictions make physical sense
"""
import pandas as pd
import numpy as np

# ----- CONFIGURATION -----
PREDICTIONS_CSV = "/Users/ninatormann/Desktop/6mMixed/predictions_4class.csv"

def validate_predictions():
    """Check if predictions follow physical rules."""
    
    print("="*70)
    print("PREDICTION VALIDATION - PHYSICS-BASED SANITY CHECKS")
    print("="*70)
    
    # Load predictions
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"\n✅ Loaded {len(df)} predictions")
    
    # Show distribution
    print(f"\n📊 Classification Distribution:")
    counts = df['classification'].value_counts()
    for cls, count in counts.items():
        pct = count / len(df) * 100
        print(f"   {cls:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Physics-based validation rules
    print(f"\n{'='*70}")
    print("PHYSICS-BASED VALIDATION CHECKS")
    print(f"{'='*70}")
    
    issues = []
    
    # CHECK 1: Overdense should have long duration
    print(f"\n1️⃣  Checking OVERDENSE predictions...")
    overdense = df[df['classification'] == 'overdense']
    
    if len(overdense) > 0:
        short_overdense = overdense[overdense['peak_duration'] < 0.3]
        
        if len(short_overdense) > 0:
            print(f"   ⚠️  Found {len(short_overdense)} overdense with SHORT duration (<0.3s)")
            print(f"   These might be mislabeled - overdense should be >0.5s")
            for idx, row in short_overdense.head(5).iterrows():
                print(f"      • {row['filename']}: {row['peak_duration']:.2f}s (conf: {row['confidence']:.1%})")
            issues.append(f"{len(short_overdense)} suspicious overdense (too short)")
        else:
            print(f"   ✅ All {len(overdense)} overdense have reasonable duration (≥0.3s)")
    else:
        print(f"   ℹ️  No overdense predictions found")
    
    # CHECK 2: Underdense should have brief duration
    print(f"\n2️⃣  Checking UNDERDENSE predictions...")
    underdense = df[df['classification'] == 'underdense']
    
    if len(underdense) > 0:
        long_underdense = underdense[underdense['peak_duration'] > 0.7]
        
        if len(long_underdense) > 0:
            print(f"   ⚠️  Found {len(long_underdense)} underdense with LONG duration (>0.7s)")
            print(f"   These might be mislabeled - underdense should be <0.5s")
            for idx, row in long_underdense.head(5).iterrows():
                print(f"      • {row['filename']}: {row['peak_duration']:.2f}s (conf: {row['confidence']:.1%})")
            issues.append(f"{len(long_underdense)} suspicious underdense (too long)")
        else:
            print(f"   ✅ All {len(underdense)} underdense have reasonable duration (≤0.7s)")
    else:
        print(f"   ℹ️  No underdense predictions found")
    
    # CHECK 3: Aircraft should have multiple peaks
    print(f"\n3️⃣  Checking AIRCRAFT predictions...")
    aircraft = df[df['classification'] == 'aircraft']
    
    if len(aircraft) > 0:
        single_peak_aircraft = aircraft[aircraft['num_peaks'] <= 1]
        
        if len(single_peak_aircraft) > 0:
            print(f"   ⚠️  Found {len(single_peak_aircraft)} aircraft with SINGLE peak")
            print(f"   These might be mislabeled - aircraft usually have 3+ bursts")
            for idx, row in single_peak_aircraft.head(5).iterrows():
                print(f"      • {row['filename']}: {row['num_peaks']} peaks (conf: {row['confidence']:.1%})")
            issues.append(f"{len(single_peak_aircraft)} suspicious aircraft (single peak)")
        else:
            print(f"   ✅ All {len(aircraft)} aircraft have multiple peaks")
        
        # Also check if aircraft has reasonable total duration
        brief_aircraft = aircraft[aircraft['time_above_threshold'] < 1.0]
        if len(brief_aircraft) > 0:
            print(f"   ⚠️  Found {len(brief_aircraft)} aircraft with brief duration (<1.0s)")
            print(f"   Aircraft usually last >2s")
    else:
        print(f"   ℹ️  No aircraft predictions found")
    
    # CHECK 4: Noise should have weak signal
    print(f"\n4️⃣  Checking NOISE predictions...")
    noise = df[df['classification'] == 'noise']
    
    if len(noise) > 0:
        strong_noise = noise[noise['peak_height_ratio'] > 0.5]
        
        if len(strong_noise) > 0:
            print(f"   ⚠️  Found {len(strong_noise)} noise with STRONG signal (ratio >0.5)")
            print(f"   These might actually be meteors - noise should be weak")
            for idx, row in strong_noise.head(5).iterrows():
                print(f"      • {row['filename']}: ratio={row['peak_height_ratio']:.2f} (conf: {row['confidence']:.1%})")
            issues.append(f"{len(strong_noise)} suspicious noise (too strong)")
        else:
            print(f"   ✅ All {len(noise)} noise have weak signals")
    else:
        print(f"   ℹ️  No noise predictions found")
    
    # CHECK 5: Low confidence predictions
    print(f"\n5️⃣  Checking CONFIDENCE levels...")
    low_conf = df[df['confidence'] < 0.5]
    
    if len(low_conf) > 0:
        print(f"   ⚠️  Found {len(low_conf)} predictions with LOW confidence (<50%)")
        print(f"   The model is uncertain about these - review manually:")
        for idx, row in low_conf.head(10).iterrows():
            print(f"      • {row['filename']}: {row['classification']} ({row['confidence']:.1%})")
    else:
        print(f"   ✅ All predictions have reasonable confidence (≥50%)")
    
    # CHECK 6: Borderline meteors (around 0.5s threshold)
    print(f"\n6️⃣  Checking BORDERLINE meteors (0.4-0.6s duration)...")
    meteors = df[df['classification'].isin(['overdense', 'underdense'])]
    borderline = meteors[(meteors['peak_duration'] >= 0.4) & (meteors['peak_duration'] <= 0.6)]
    
    if len(borderline) > 0:
        print(f"   ℹ️  Found {len(borderline)} borderline meteors near 0.5s threshold")
        print(f"   These are naturally ambiguous - check if classifications make sense:")
        for idx, row in borderline.head(5).iterrows():
            print(f"      • {row['filename']}: {row['classification']} ({row['peak_duration']:.2f}s, conf: {row['confidence']:.1%})")
    else:
        print(f"   ✅ No borderline cases - clean separation")
    
    # SUMMARY
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    if len(issues) == 0:
        print(f"\n✅ ALL CHECKS PASSED!")
        print(f"   Predictions follow physical rules")
        print(f"   Model appears to be working correctly")
    else:
        print(f"\n⚠️  Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"   • {issue}")
        
        print(f"\n💡 What to do about issues:")
        print(f"   1. Review the suspicious files listed above")
        print(f"   2. Visualize them with visualize_ml_predictions.py")
        print(f"   3. Check if they're at class boundaries (0.4-0.6s)")
        print(f"   4. Low confidence predictions are naturally uncertain")
        print(f"\n   If most look reasonable, the model is working!")
    
    # Confidence statistics
    print(f"\n📊 Confidence Statistics:")
    print(f"   Mean confidence: {df['confidence'].mean():.1%}")
    print(f"   Median confidence: {df['confidence'].median():.1%}")
    
    high_conf = df[df['confidence'] >= 0.8]
    med_conf = df[(df['confidence'] >= 0.6) & (df['confidence'] < 0.8)]
    low_conf = df[df['confidence'] < 0.6]
    
    print(f"\n   High confidence (≥80%): {len(high_conf)} ({len(high_conf)/len(df)*100:.1f}%)")
    print(f"   Medium confidence (60-80%): {len(med_conf)} ({len(med_conf)/len(df)*100:.1f}%)")
    print(f"   Low confidence (<60%): {len(low_conf)} ({len(low_conf)/len(df)*100:.1f}%)")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    total_suspicious = sum([len(df[df['classification'] == 'overdense'][df['peak_duration'] < 0.3]),
                           len(df[df['classification'] == 'underdense'][df['peak_duration'] > 0.7]),
                           len(df[df['classification'] == 'noise'][df['peak_height_ratio'] > 0.5])])
    
    if total_suspicious < len(df) * 0.05:  # Less than 5% suspicious
        print(f"\n✅ EXCELLENT: <5% suspicious predictions ({total_suspicious}/{len(df)})")
        print(f"   Your model is working very well!")
        print(f"\n   Next steps:")
        print(f"   ✅ Use high-confidence predictions (≥80%) for analysis")
        print(f"   ✅ Manually review borderline cases if needed")
        print(f"   ✅ Trust the model for bulk classification")
    elif total_suspicious < len(df) * 0.15:  # 5-15% suspicious
        print(f"\n✅ GOOD: 5-15% suspicious predictions ({total_suspicious}/{len(df)})")
        print(f"   Model is working reasonably well")
        print(f"\n   Suggestions:")
        print(f"   • Review suspicious cases listed above")
        print(f"   • Focus on high-confidence predictions")
        print(f"   • Some errors are expected at class boundaries")
    else:  # >15% suspicious
        print(f"\n⚠️  REVIEW NEEDED: >15% suspicious predictions ({total_suspicious}/{len(df)})")
        print(f"\n   This might indicate:")
        print(f"   • Training data labeling errors")
        print(f"   • Model needs more training data")
        print(f"   • Features not extracting correctly during prediction")
        print(f"\n   Action items:")
        print(f"   1. Review training labels for consistency")
        print(f"   2. Check if audio files are from same source as training")
        print(f"   3. Visualize suspicious predictions to understand errors")


if __name__ == "__main__":
    validate_predictions()
