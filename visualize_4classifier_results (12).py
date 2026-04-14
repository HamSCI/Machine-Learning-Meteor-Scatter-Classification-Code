#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Rule-Based Classification Results
Creates labeled plots for overdense and underdense meteors
Designed to work with classify_simple_RULEBASED.py output
"""
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# ----- CONFIGURATION -----
CLASSIFICATION_CSV = "/Users/ninatormann/Desktop/10mMixed/simple_classification.csv"
AUDIO_DIR = "/Users/ninatormann/Desktop/10mMixed"
OUTPUT_DIR = "/Users/ninatormann/Desktop/10mMixed/9density_plots"

# Plotting settings
MAX_PLOTS_PER_TYPE = 7   # Max number of plots per category (overdense/underdense)
MAX_AIRCRAFT_PLOTS = 3   # Max aircraft examples
MAX_NOISE_PLOTS = 3      # Max noise examples
MIN_CONFIDENCE = 'medium'  # 'high', 'medium', or 'low'

# Audio analysis settings
FREQ_MIN = 500
FREQ_MAX = 2500
WINDOW_DURATION = 0.072


def compute_amplitude_timeseries(data, fs):
    """Compute MSK144 amplitude over time."""
    window_size = int(fs * WINDOW_DURATION)
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
        
        mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
        avg_amp = np.mean(amps[mask])
        
        times.append(i * WINDOW_DURATION)
        amplitudes.append(avg_amp)
    
    return np.array(times), np.array(amplitudes)


def create_density_plot(filename, audio_dir, signal_type, output_dir):
    """Create visualization plot for a single file."""
    
    try:
        filepath = os.path.join(audio_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found: {filepath}")
            return False
        
        # Read audio file
        fs, data = wavfile.read(filepath)
        
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(data.dtype)
        
        # Compute spectrogram
        f, t, Sxx = spectrogram(data, fs=fs, nperseg=2048, noverlap=1024)
        freq_mask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
        power_db = 10 * np.log10(Sxx[freq_mask, :] + 1e-12)
        
        # Compute amplitude timeseries
        times, amplitudes = compute_amplitude_timeseries(data, fs)
        
        # Detection threshold
        baseline = np.median(amplitudes)
        threshold = baseline * 1.35
        above_threshold = amplitudes > threshold
        
        # Create figure with 2 subplots - MADE BIGGER
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))  # Increased from 14x7 to 16x8
        
        # Color scheme and description based on signal type
        if signal_type == 'overdense':
            title_color = 'blue'
            highlight_color = 'blue'
            description = 'Long-duration, dense ionization trail'
        elif signal_type == 'underdense':
            title_color = 'red'
            highlight_color = 'red'
            description = 'Brief, sparse ionization trail'
        elif signal_type == 'aircraft':
            title_color = 'green'
            highlight_color = 'green'
            description = 'Multiple bursts from aircraft scatter'
        elif signal_type == 'noise':
            title_color = 'gray'
            highlight_color = 'gray'
            description = 'Weak signal or interference'
        else:
            title_color = 'black'
            highlight_color = 'black'
            description = ''
        
        # Main title
        fig.suptitle(f'{signal_type.upper()}\n{description}', 
                     fontsize=16, fontweight='bold', color=title_color, x=0.5, ha='center')
        
        # Plot 1: Spectrogram
        im = ax1.pcolormesh(t, f[freq_mask], power_db, shading='auto', cmap='plasma', vmin=0, vmax=30)
        ax1.axhline(FREQ_MIN, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax1.axhline(FREQ_MAX, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax1.fill_between(t, FREQ_MIN, FREQ_MAX, color=highlight_color, alpha=0.15)
        ax1.set_ylabel('Frequency (Hz)', fontsize=11)
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_title(f'Spectrogram (MSK144 Range: {FREQ_MIN}-{FREQ_MAX} Hz)', fontsize=12)
        ax1.set_ylim([FREQ_MIN, FREQ_MAX])
        ax1.set_xlim([0.5, 14])  # Start at 0.5s to make data pop
        
        # Plot 2: Amplitude over time
        ax2.plot(times, amplitudes, color='darkred', linewidth=1.5, label=f'MSK144 Amplitude ({FREQ_MIN}-{FREQ_MAX} Hz)')
        ax2.axhline(threshold, color='orange', linestyle='--', linewidth=2, 
                   label=f'Detection Threshold (35% above baseline)')
        ax2.axhline(baseline, color='gray', linestyle=':', linewidth=1.5, 
                   label=f'Baseline: {baseline:.0f}')
        
        # Highlight detected signal
        if np.any(above_threshold):
            ax2.fill_between(times, 0, amplitudes, where=above_threshold, 
                           color=highlight_color, alpha=0.3, 
                           label=f'Detected signal ({np.sum(above_threshold)*WINDOW_DURATION:.2f}s duration)')
        
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Amplitude (relative)', fontsize=11)
        ax2.set_title(f'Amplitude Analysis - {signal_type.capitalize()} Characteristics', fontsize=12)
        ax2.set_xlim([0.5, 14])  # Start at 0.5s to make data pop
        ax2.grid(True, alpha=0.3)
        
        # Move legend outside plot area - aligned with TOP of plot
        ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.9)
        
        # Info box - MOVED DOWN, aligned with middle/bottom
        peak_duration = np.sum(above_threshold) * WINDOW_DURATION
        num_peaks = np.sum(np.diff(above_threshold.astype(int)) == 1)
        peak_height = np.max(amplitudes) - baseline
        
        info_text = (
            f"Classification:\n{signal_type.upper()}\n\n"
            f"Peak Duration:\n{peak_duration:.2f}s\n\n"
            f"Number of Peaks:\n{num_peaks}\n\n"
            f"Peak Height:\n{peak_height:.0f}"
        )
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(1.03, 0.25, info_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='center', bbox=props)  # Changed from 1.02, 0.35 to 1.03, 0.25
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.85, 0.94])
        
        # Add colorbar AFTER tight_layout - positioned at 0.63
        cbar_ax = fig.add_axes([0.63, 0.50, 0.02, 0.35])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Power (dB)', fontsize=10)
        
        # Save plot
        plot_filename = f'{signal_type}_{filename.replace(".wav", "")}_plot.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Error plotting {filename}: {e}")
        return False


def main():
    print("="*70)
    print("VISUALIZING RULE-BASED CLASSIFICATION RESULTS")
    print("="*70)
    
    # Check classification CSV exists
    if not os.path.exists(CLASSIFICATION_CSV):
        print(f"\n❌ Error: Classification CSV not found: {CLASSIFICATION_CSV}")
        print("Please run classify_simple_RULEBASED.py first!")
        return
    
    # Load classifications
    print(f"\n📊 Loading classifications from: {CLASSIFICATION_CSV}")
    df = pd.read_csv(CLASSIFICATION_CSV)
    print(f"✅ Loaded {len(df)} classifications")
    
    # Normalize column names - handle both formats
    # Rule-based uses: max_burst_duration_s, total_duration_s
    # ML uses: peak_duration, time_above_threshold
    if 'peak_duration' in df.columns and 'max_burst_duration_s' not in df.columns:
        df['max_burst_duration_s'] = df['peak_duration']
    if 'time_above_threshold' in df.columns and 'total_duration_s' not in df.columns:
        df['total_duration_s'] = df['time_above_threshold']
    
    # Show summary
    print(f"\n📈 Classification Summary:")
    counts = df['classification'].value_counts()
    for cls, count in counts.items():
        pct = count / len(df) * 100
        print(f"   {cls.capitalize()}: {count} ({pct:.1f}%)")
    
    # Filter to only meteors (overdense and underdense)
    meteors = df[df['classification'].isin(['overdense', 'underdense'])].copy()
    print(f"\n🌠 {len(meteors)} meteor detections (excluding aircraft and noise)")
    
    # Handle confidence format - works with BOTH text and numeric
    if 'confidence' in meteors.columns:
        # Check if confidence is text or numeric
        sample_conf = meteors['confidence'].iloc[0] if len(meteors) > 0 else None
        
        if isinstance(sample_conf, str):
            # Text confidence from rule-based classifier ('high', 'medium', 'low')
            print(f"   Detected text confidence format (rule-based)")
            confidence_rank = {'high': 3, 'medium': 2, 'low': 1}
            min_conf_rank = confidence_rank.get(MIN_CONFIDENCE, 2)
            meteors['conf_rank'] = meteors['confidence'].map(confidence_rank)
            high_conf = meteors[meteors['conf_rank'] >= min_conf_rank].copy()
        else:
            # Numeric confidence from ML model (0.0-1.0)
            print(f"   Detected numeric confidence format (ML model)")
            conf_threshold = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
            min_conf_value = conf_threshold.get(MIN_CONFIDENCE, 0.6)
            high_conf = meteors[meteors['confidence'] >= min_conf_value].copy()
            # Create ranking for sorting (higher confidence = higher rank)
            high_conf['conf_rank'] = high_conf['confidence']
    else:
        print(f"   No confidence column found - using all meteors")
        high_conf = meteors.copy()
        high_conf['conf_rank'] = 1.0
    
    print(f"   {len(high_conf)} with ≥{MIN_CONFIDENCE} confidence")
    
    if len(high_conf) == 0:
        print(f"\n⚠️  No meteors found with confidence ≥ {MIN_CONFIDENCE}")
        print(f"   Try lowering MIN_CONFIDENCE in the script")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 Plots will be saved to: {OUTPUT_DIR}")
    
    # Plot overdense examples
    print(f"\n{'='*70}")
    print("PLOTTING OVERDENSE EXAMPLES")
    print(f"{'='*70}")
    
    overdense = high_conf[high_conf['classification'] == 'overdense'].sort_values(
        'conf_rank', ascending=False
    ).head(MAX_PLOTS_PER_TYPE)
    
    overdense_count = 0
    for idx, row in overdense.iterrows():
        print(f"\n🔵 Processing: {row['filename']} (confidence: {row['confidence']}, duration: {row['max_burst_duration_s']:.2f}s)")
        success = create_density_plot(
            row['filename'], 
            AUDIO_DIR, 
            'overdense',
            OUTPUT_DIR
        )
        if success:
            overdense_count += 1
    
    print(f"\n✅ Created {overdense_count} overdense plots")
    
    # Plot underdense examples
    print(f"\n{'='*70}")
    print("PLOTTING UNDERDENSE EXAMPLES")
    print(f"{'='*70}")
    
    underdense = high_conf[high_conf['classification'] == 'underdense'].sort_values(
        'conf_rank', ascending=False
    ).head(MAX_PLOTS_PER_TYPE)
    
    underdense_count = 0
    for idx, row in underdense.iterrows():
        print(f"\n🔴 Processing: {row['filename']} (confidence: {row['confidence']}, duration: {row['max_burst_duration_s']:.2f}s)")
        success = create_density_plot(
            row['filename'], 
            AUDIO_DIR, 
            'underdense',
            OUTPUT_DIR
        )
        if success:
            underdense_count += 1
    
    print(f"\n✅ Created {underdense_count} underdense plots")
    
    # Plot aircraft examples
    print(f"\n{'='*70}")
    print("PLOTTING AIRCRAFT SCATTER EXAMPLES")
    print(f"{'='*70}")
    
    aircraft_all = df[df['classification'] == 'aircraft']
    print(f"   Found {len(aircraft_all)} aircraft detections in CSV")
    
    if len(aircraft_all) > 0:
        # Sort by confidence (works for both numeric and will skip if text)
        if 'confidence' in aircraft_all.columns:
            sample_conf = aircraft_all['confidence'].iloc[0]
            if isinstance(sample_conf, (int, float)):
                aircraft = aircraft_all.sort_values('confidence', ascending=False).head(MAX_AIRCRAFT_PLOTS)
            else:
                # Text confidence - sort by num_bursts instead
                aircraft = aircraft_all.sort_values('num_bursts', ascending=False).head(MAX_AIRCRAFT_PLOTS)
        else:
            aircraft = aircraft_all.head(MAX_AIRCRAFT_PLOTS)
        
        print(f"   Selecting top {len(aircraft)} for plotting...")
        
        aircraft_count = 0
        for idx, row in aircraft.iterrows():
            conf_str = f"{row['confidence']:.2f}" if isinstance(row.get('confidence'), (int, float)) else row.get('confidence', 'N/A')
            print(f"\n✈️  Processing: {row['filename']} (confidence: {conf_str}, bursts: {row['num_bursts']})")
            success = create_density_plot(
                row['filename'], 
                AUDIO_DIR, 
                'aircraft',
                OUTPUT_DIR
            )
            if success:
                aircraft_count += 1
            else:
                print(f"     ❌ Failed to create plot")
        print(f"\n✅ Created {aircraft_count} aircraft plots")
    else:
        print(f"\n   ⚠️  No aircraft scatter detected in dataset")
    
    # Plot noise examples
    print(f"\n{'='*70}")
    print("PLOTTING NOISE/INTERFERENCE EXAMPLES")
    print(f"{'='*70}")
    
    noise_all = df[df['classification'] == 'noise']
    print(f"   Found {len(noise_all)} noise detections in CSV")
    
    if len(noise_all) > 0:
        # Sort by peak_height_ratio (lower = weaker signal)
        noise = noise_all.sort_values('peak_height_ratio', ascending=True).head(MAX_NOISE_PLOTS)
        print(f"   Selecting top {len(noise)} for plotting...")
        
        noise_count = 0
        for idx, row in noise.iterrows():
            conf_str = f"{row['confidence']:.2f}" if isinstance(row.get('confidence'), (int, float)) else row.get('confidence', 'N/A')
            print(f"\n📡 Processing: {row['filename']} (confidence: {conf_str}, peak ratio: {row['peak_height_ratio']:.2f})")
            success = create_density_plot(
                row['filename'], 
                AUDIO_DIR, 
                'noise',
                OUTPUT_DIR
            )
            if success:
                noise_count += 1
            else:
                print(f"     ❌ Failed to create plot")
        print(f"\n✅ Created {noise_count} noise plots")
    else:
        print(f"\n   ⚠️  No noise/interference detected in dataset")
    
    # Final summary
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"\n✅ Total plots created: {overdense_count + underdense_count + aircraft_count + noise_count}")
    print(f"   🔵 Overdense: {overdense_count}")
    print(f"   🔴 Underdense: {underdense_count}")
    print(f"   ✈️  Aircraft: {aircraft_count}")
    print(f"   📡 Noise: {noise_count}")
    print(f"\n📁 All plots saved to: {OUTPUT_DIR}")
    print(f"\n💡 These examples show the clear differences between:")
    print(f"   • Meteor scatter (overdense vs underdense)")
    print(f"   • Aircraft scatter (multiple bursts)")
    print(f"   • Noise/interference (weak signals)")



if __name__ == "__main__":
    main()
