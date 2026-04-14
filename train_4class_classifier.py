#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 4-Class Meteor Classifier: Overdense, Underdense, Aircraft, Noise
Handles class imbalance with balanced weights
"""
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# ----- CONFIGURATION -----
TRAINING_CSV = "/Users/ninatormann/Desktop/10mMixed/simple_classification.csv"
AUDIO_DIR = "/Users/ninatormann/Desktop/10mMixed"
OUTPUT_DIR = "/Users/ninatormann/Desktop/10mMixed"

# Model output
MODEL_FILE = "meteor_4class_classifier.pkl"
FEATURES_CSV = "training_features_4class.csv"

# Feature extraction settings
FREQ_MIN = 500
FREQ_MAX = 2000
WINDOW_DURATION = 0.072


class MeteorFeatureExtractor:
    """Extract features from audio files."""
    
    def __init__(self, freq_min=500, freq_max=2000, window_duration=0.072):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.window_duration = window_duration
    
    def compute_amplitude_timeseries(self, data, fs):
        """Compute MSK144 amplitude over time."""
        window_size = int(fs * self.window_duration)
        num_windows = len(data) // window_size
        
        times = []
        amplitudes = []
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            segment = data[start:end]
            
            fft_result = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), d=1/fs)
            amps = np.abs(fft_result)
            
            mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
            avg_amp = np.mean(amps[mask])
            
            times.append(i * self.window_duration)
            amplitudes.append(avg_amp)
        
        return np.array(times), np.array(amplitudes)
    
    def extract_features(self, filepath):
        """Extract comprehensive features from a WAV file."""
        try:
            # Read audio file
            fs, data = wavfile.read(filepath)
            
            if data.ndim > 1:
                data = np.mean(data, axis=1).astype(data.dtype)
            
            # Compute amplitude timeseries
            times, amplitudes = self.compute_amplitude_timeseries(data, fs)
            
            # Basic statistics
            amp_mean = np.mean(amplitudes)
            amp_median = np.median(amplitudes)
            amp_std = np.std(amplitudes)
            amp_max = np.max(amplitudes)
            amp_min = np.min(amplitudes)
            amp_range = amp_max - amp_min
            
            # Peak detection
            threshold = amp_median * 1.35
            above_threshold = amplitudes > threshold
            num_peaks = np.sum(np.diff(above_threshold.astype(int)) == 1)
            
            # Time above threshold
            time_above_threshold = np.sum(above_threshold) * self.window_duration
            percent_above_threshold = np.sum(above_threshold) / len(amplitudes) * 100
            
            # Peak characteristics
            if np.any(above_threshold):
                peak_height_ratio = (amp_max - amp_median) / amp_median
                peak_indices = np.where(above_threshold)[0]
                peak_duration = len(peak_indices) * self.window_duration
                
                # Rise and fall time
                first_peak_idx = peak_indices[0]
                last_peak_idx = peak_indices[-1]
                
                if first_peak_idx > 0:
                    rise_time = first_peak_idx * self.window_duration
                    rise_rate = (amp_max - amp_median) / (rise_time + 1e-10)
                else:
                    rise_time = 0
                    rise_rate = 0
                
                if last_peak_idx < len(amplitudes) - 1:
                    fall_time = (len(amplitudes) - last_peak_idx) * self.window_duration
                    fall_rate = (amp_max - amp_median) / (fall_time + 1e-10)
                else:
                    fall_time = 0
                    fall_rate = 0
            else:
                peak_height_ratio = 0
                peak_duration = 0
                rise_time = 0
                rise_rate = 0
                fall_time = 0
                fall_rate = 0
            
            # Variability features
            coefficient_of_variation = amp_std / amp_mean * 100 if amp_mean > 0 else 0
            
            # Distribution shape
            amp_skewness = skew(amplitudes)
            amp_kurtosis = kurtosis(amplitudes)
            
            # Temporal patterns
            amp_diff = np.diff(amplitudes)
            max_rise = np.max(amp_diff) if len(amp_diff) > 0 else 0
            max_fall = abs(np.min(amp_diff)) if len(amp_diff) > 0 else 0
            avg_change = np.mean(np.abs(amp_diff)) if len(amp_diff) > 0 else 0
            
            # Spectral features
            f, t, Sxx = spectrogram(data, fs=fs, nperseg=2048, noverlap=1024)
            freq_mask = (f >= self.freq_min) & (f <= self.freq_max)
            power_db = 10 * np.log10(Sxx[freq_mask, :] + 1e-12)
            
            spectral_mean = np.mean(power_db)
            spectral_max = np.max(power_db)
            spectral_std = np.std(power_db)
            
            # Compile features
            features = {
                'amp_mean': amp_mean,
                'amp_median': amp_median,
                'amp_std': amp_std,
                'amp_max': amp_max,
                'amp_range': amp_range,
                'coefficient_of_variation': coefficient_of_variation,
                'num_peaks': num_peaks,
                'peak_height_ratio': peak_height_ratio,
                'peak_duration': peak_duration,
                'time_above_threshold': time_above_threshold,
                'percent_above_threshold': percent_above_threshold,
                'rise_time': rise_time,
                'fall_time': fall_time,
                'rise_rate': rise_rate,
                'fall_rate': fall_rate,
                'max_rise': max_rise,
                'max_fall': max_fall,
                'avg_change': avg_change,
                'amp_skewness': amp_skewness,
                'amp_kurtosis': amp_kurtosis,
                'spectral_mean': spectral_mean,
                'spectral_max': spectral_max,
                'spectral_std': spectral_std,
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None


def main():
    print("="*70)
    print("TRAINING 4-CLASS METEOR CLASSIFIER")
    print("Classes: Overdense, Underdense, Aircraft, Noise")
    print("="*70)
    
    # Load training data
    if not os.path.exists(TRAINING_CSV):
        print(f"\n❌ Error: Training CSV not found: {TRAINING_CSV}")
        return
    
    print(f"\n📊 Loading training data from: {TRAINING_CSV}")
    df = pd.read_csv(TRAINING_CSV)
    print(f"✅ Loaded {len(df)} labeled files")
    
    # Show class distribution
    print(f"\n📈 Class Distribution:")
    counts = df['classification'].value_counts()
    for cls, count in counts.items():
        pct = count / len(df) * 100
        print(f"   {cls.capitalize():15s}: {count:4d} ({pct:5.1f}%)")
    
    # Initialize feature extractor
    extractor = MeteorFeatureExtractor()
    
    # Extract features for all files
    print(f"\n🔍 Extracting features from audio files...")
    
    features_list = []
    labels_list = []
    filenames_list = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"   Progress: {idx + 1}/{len(df)}")
        
        filepath = os.path.join(AUDIO_DIR, row['filename'])
        
        if not os.path.exists(filepath):
            print(f"   ⚠️  File not found: {row['filename']}")
            continue
        
        features = extractor.extract_features(filepath)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(row['classification'])
            filenames_list.append(row['filename'])
    
    print(f"\n✅ Extracted features from {len(features_list)} files")
    
    # Create features dataframe
    features_df = pd.DataFrame(features_list)
    features_df['filename'] = filenames_list
    features_df['classification'] = labels_list
    
    # Save features
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    features_path = os.path.join(OUTPUT_DIR, FEATURES_CSV)
    features_df.to_csv(features_path, index=False)
    print(f"💾 Saved features to: {features_path}")
    
    # Prepare data for training
    feature_columns = [col for col in features_df.columns if col not in ['filename', 'classification']]
    
    X = features_df[feature_columns].values
    y = features_df['classification'].values
    
    print(f"\n🔧 Preparing ML model...")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Samples: {len(X)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing:  {len(X_test)} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with class balancing
    print(f"\n🤖 Training Random Forest Classifier...")
    print(f"   Using class_weight='balanced' to handle imbalanced data")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # CRITICAL for handling imbalance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    print(f"✅ Model trained!")
    
    # Evaluate on test set
    print(f"\n📊 Test Set Performance:")
    y_pred = model.predict(X_test_scaled)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"   Overall Accuracy: {accuracy:.1%}")
    
    # Per-class metrics
    print(f"\n   Detailed Classification Report:")
    report = classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_,
        digits=3
    )
    print(report)
    
    # Confusion matrix
    print(f"\n   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"   {'':15s}", end="")
    for cls in label_encoder.classes_:
        print(f"{cls[:10]:>10s}", end="")
    print()
    
    for i, cls in enumerate(label_encoder.classes_):
        print(f"   {cls:15s}", end="")
        for j in range(len(label_encoder.classes_)):
            print(f"{cm[i,j]:10d}", end="")
        print()
    
    # Cross-validation
    print(f"\n🔄 5-Fold Cross-Validation:")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"   CV Scores: {[f'{s:.1%}' for s in cv_scores]}")
    print(f"   Mean: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    
    # Feature importance
    print(f"\n🔍 Top 10 Most Important Features:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_columns))):
        idx = indices[i]
        print(f"   {i+1}. {feature_columns[idx]:30s} : {importances[idx]:.4f}")
    
    # Package model
    print(f"\n📦 Packaging model...")
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': feature_columns,
        'training_info': {
            'n_samples': len(X),
            'n_features': len(feature_columns),
            'classes': list(label_encoder.classes_),
            'test_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    }
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, MODEL_FILE)
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"💾 Model saved to: {model_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n✅ 4-Class Classifier Ready:")
    print(f"   📊 Trained on {len(X)} samples")
    print(f"   🎯 Test Accuracy: {accuracy:.1%}")
    print(f"   🔄 CV Score: {cv_scores.mean():.1%}")
    print(f"\n💡 Model can now classify:")
    print(f"   • Overdense meteors")
    print(f"   • Underdense meteors")
    print(f"   • Aircraft scatter")
    print(f"   • Noise/interference")
    print(f"\n📁 Next step: Use predict_4class.py to classify new files!")


if __name__ == "__main__":
    main()
