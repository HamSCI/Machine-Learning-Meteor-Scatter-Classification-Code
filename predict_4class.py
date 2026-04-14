#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict with 4-Class Meteor Classifier
Classifies files as: Overdense, Underdense, Aircraft, or Noise
"""
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.stats import kurtosis, skew
import pickle

# ----- CONFIGURATION -----
MODEL_FILE = "/Users/ninatormann/Desktop/10mMixed/meteor_4class_classifier.pkl"
INPUT_DIR = "/Users/ninatormann/Desktop/6mMixed"
OUTPUT_CSV = "/Users/ninatormann/Desktop/6mMixed/predictions_4class.csv"

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
    print("4-CLASS METEOR CLASSIFICATION")
    print("Overdense | Underdense | Aircraft | Noise")
    print("="*70)
    
    # Load model
    if not os.path.exists(MODEL_FILE):
        print(f"\n❌ Error: Model file not found: {MODEL_FILE}")
        print("Please run train_4class_classifier.py first!")
        return
    
    print(f"\n📦 Loading model: {MODEL_FILE}")
    with open(MODEL_FILE, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoder = model_package['label_encoder']
    feature_columns = model_package['feature_columns']
    
    print(f"✅ Model loaded successfully")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Check input directory
    if not os.path.exists(INPUT_DIR):
        print(f"\n❌ Error: Input directory not found: {INPUT_DIR}")
        return
    
    # Get WAV files
    wav_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.wav')])
    
    if len(wav_files) == 0:
        print(f"\n❌ No WAV files found in: {INPUT_DIR}")
        return
    
    print(f"\n🔍 Found {len(wav_files)} files to classify")
    
    # Extract features and predict
    extractor = MeteorFeatureExtractor()
    results = []
    
    print(f"\nProcessing...")
    for i, filename in enumerate(wav_files, 1):
        if i % 50 == 0 or i == len(wav_files):
            print(f"  Progress: {i}/{len(wav_files)} ({i/len(wav_files)*100:.1f}%)")
        
        filepath = os.path.join(INPUT_DIR, filename)
        
        try:
            # Extract features
            features = extractor.extract_features(filepath)
            
            if features is None:
                continue
            
            # Prepare for prediction
            features_df = pd.DataFrame([features])
            X = features_df[feature_columns].values
            X_scaled = scaler.transform(X)
            
            # Predict
            prediction_encoded = model.predict(X_scaled)[0]
            classification = label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)[0]
                confidence = probabilities[prediction_encoded]
            else:
                confidence = 1.0
            
            # Store result
            results.append({
                'filename': filename,
                'classification': classification,
                'confidence': round(confidence, 3),
                'peak_duration': round(features['peak_duration'], 3),
                'num_peaks': int(features['num_peaks']),
                'time_above_threshold': round(features['time_above_threshold'], 3),
                'peak_height_ratio': round(features['peak_height_ratio'], 3),
            })
            
        except Exception as e:
            print(f"  ⚠️  Error processing {filename}: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    # Summary
    print(f"\n{'='*70}")
    print("CLASSIFICATION COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n✅ Classified {len(results_df)} files")
    print(f"📄 Results saved to: {OUTPUT_CSV}")
    
    # Show breakdown
    print(f"\n📊 Classification Summary:")
    
    counts = results_df['classification'].value_counts()
    
    for classification, count in counts.items():
        pct = count / len(results_df) * 100
        
        if classification == 'overdense':
            icon = '🔵'
            desc = 'Overdense meteor'
        elif classification == 'underdense':
            icon = '🔴'
            desc = 'Underdense meteor'
        elif classification == 'aircraft':
            icon = '✈️ '
            desc = 'Aircraft scatter'
        elif classification == 'noise':
            icon = '📡'
            desc = 'Noise/interference'
        else:
            icon = '❓'
            desc = ''
        
        print(f"   {icon} {classification.capitalize():15s}: {count:4d} ({pct:5.1f}%) - {desc}")
    
    # Show meteor-only stats
    meteors = results_df[results_df['classification'].isin(['overdense', 'underdense'])]
    print(f"\n🌠 Meteor Detections: {len(meteors)} total")
    if len(meteors) > 0:
        meteor_counts = meteors['classification'].value_counts()
        for cls, cnt in meteor_counts.items():
            pct = cnt / len(meteors) * 100
            print(f"   {cls.capitalize()}: {cnt} ({pct:.1f}%)")
    
    print(f"\n💡 Use visualize_rulebased_results.py to create plots!")


if __name__ == "__main__":
    main()
