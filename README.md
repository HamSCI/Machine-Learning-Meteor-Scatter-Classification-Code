# Machine-Learning-Meteor-Scatter-Classification-Code
Codes used for training, classifying and visualizing meteor scatter data
# Machine-Learning-Meteor-Scatter-Classification-Code

Automated classification of meteor scatter signals from amateur radio WAV recordings using a Random Forest machine learning classifier.

Developed by Nina Tormann as part of a Bachelor of Science thesis at the University of Scranton, Department of Physics and Engineering, under the supervision of Dr. Nathaniel Frissell W2NAF.

---

## Overview

This repository contains the full pipeline for training, running, validating, and visualizing a four-class machine learning classifier applied to meteor scatter audio recordings collected during HamSCI Meteor Scatter QSO Party (MSQP) events. The classifier distinguishes between:

- **Underdense meteor scatter** — brief, sparse ionization trail reflections (typically < 0.5 seconds)
- **Overdense meteor scatter** — sustained, dense plasma column reflections (typically > 0.5 seconds)
- **Aircraft scatter** — periodic multi-burst interference from aircraft reflections
- **Noise / other propagation** — background noise, sporadic-E, and other non-meteor signals

Input data is WAV audio files recorded by amateur radio operators using WSJT-X software in MSK144 mode on the 6-meter (50 MHz) and 10-meter (28 MHz) amateur radio bands. Data was collected during the 2025 Perseids (August) and Geminids (December) meteor showers and archived on [Zenodo](https://zenodo.org).

---

## Repository Structure

```
├── train_4class_classifier.py       # Phase 1: Train the Random Forest model on labeled data
├── predict_4class.py                # Phase 2: Classify new unlabeled WAV files
├── test_4class_model.py             # Evaluate model performance on a test set
├── validate_predictions.py          # Validate classifier output against known labels
├── visualize_4classifier_results.py # Generate spectrograms and result plots
├── LICENSE                          # GPL-3.0 license
└── README.md                        # This file
```

---

## Requirements

Python 3.8+ is recommended. Install dependencies with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib pickle5
```

### Full dependency list

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations and array handling |
| `pandas` | CSV I/O and data management |
| `scipy` | WAV file reading, FFT, spectrogram computation |
| `scikit-learn` | Random Forest classifier, scaling, metrics |
| `matplotlib` | Spectrogram and results visualization |
| `pickle` | Model serialization (built-in) |

---

## Data Format

### Input: WAV files
- Format: `.wav` audio files, mono or stereo (stereo is averaged to mono automatically)
- Source: WSJT-X with MSK144 mode, "Save Decoded" option enabled
- Each file corresponds to one 14-second MSK144 observation window
- Frequency range of interest: **500–2000 Hz** (MSK144 audio range)

### Training CSV
The training phase requires a CSV file with two columns:

```
filename,classification
signal_001.wav,underdense
signal_002.wav,overdense
signal_003.wav,aircraft
signal_004.wav,noise
```

Valid classification labels: `underdense`, `overdense`, `aircraft`, `noise`

---

## Usage

### Phase 1 — Train the classifier

Edit the configuration block at the top of `train_4class_classifier.py`:

```python
TRAINING_CSV = "/path/to/your/simple_classification.csv"   # labeled training data
AUDIO_DIR    = "/path/to/your/wav/files"                   # folder containing WAV files
OUTPUT_DIR   = "/path/to/output"                           # where model and features are saved
```

Then run:

```bash
python train_4class_classifier.py
```

This will:
1. Load labeled WAV files from the CSV
2. Extract 23 signal features from each file
3. Train a Random Forest classifier (100 trees, max depth 15, class-balanced)
4. Run 5-fold cross-validation
5. Print a classification report and confusion matrix
6. Save the trained model as `meteor_4class_classifier.pkl`

### Phase 2 — Classify new files

Edit the input/output paths at the top of `predict_4class.py`, then run:

```bash
python predict_4class.py
```

Each WAV file is processed individually. Results are saved to a CSV with columns for filename, predicted class, and confidence score (fraction of trees in agreement).

### Validate predictions

```bash
python validate_predictions.py
```

Compares classifier output against a ground-truth labeled set and reports accuracy metrics.

### Visualize results

```bash
python visualize_4classifier_results.py
```

Generates spectrogram plots and amplitude analysis figures for classified events, color-coded by signal type.

---

## Feature Extraction

The classifier extracts **23 quantitative features** from each WAV file, grouped into five categories:

| Category | Features |
|---|---|
| **Amplitude statistics** | mean, median, standard deviation, maximum, range |
| **Temporal dynamics** | rise time, fall time, rise rate, fall rate, max rise, max fall, avg change |
| **Peak detection** | number of peaks, peak height ratio, peak duration, time above threshold, % above threshold |
| **Distribution shape** | skewness, kurtosis, coefficient of variation |
| **Spectral** | spectral mean (dB), spectral maximum (dB), spectral standard deviation (dB) |

The detection threshold is set at **35% above the median baseline amplitude** of each recording.

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Number of trees | 100 |
| Maximum tree depth | 15 |
| Class weighting | `balanced` (compensates for noise-heavy datasets) |
| Train/test split | 75% / 25% |
| Cross-validation | 5-fold stratified |
| Feature scaling | StandardScaler (zero mean, unit variance) |
| Model format | Python `pickle` (.pkl) |

The trained model and scaler are packaged together in a single `.pkl` file so that the same scaling parameters used during training are automatically applied during classification.

---

## Results

Applied to 17,299 recordings from the 2025 Perseids and Geminids MSQP events across four datasets (Perseids 6m, Perseids 10m, Geminids 6m, Geminids 10m), the classifier achieved a **mean prediction confidence of 96.4%**.

Key findings:
- Underdense trails: mean signal duration 0.295–0.393 seconds
- Overdense trails: mean signal duration 1.370–1.678 seconds (3.6–4.9× longer)
- The 10-meter band (28 MHz) consistently showed higher overdense-to-underdense ratios than the 6-meter band (50 MHz), consistent with the lower critical electron density threshold at 28 MHz

Full results are reported in:

> Tormann, N. (2026). *Analyzing Meteor Scatter Communications Through Citizen Science and Data-Driven Methods*. Bachelor of Science Thesis, Department of Physics and Engineering, University of Scranton. Supervisor: Dr. Nathaniel Frissell.

---

## Related Data

Audio recordings and station metadata from the 2025 HamSCI Meteor Scatter QSO Parties are archived on Zenodo. All individual station submissions can be found in the [Ham Radio Science Citizen Investigation Zenodo community](https://zenodo.org/communities/hamsci/).

For information on participating in future Meteor Scatter QSO Parties or submitting data, visit: [https://hamsci.org/msqp](https://hamsci.org/msqp)

---

## Acknowledgements

This research would not have been possible without the guidance and support of many individuals across the amateur radio and scientific communities. The author would like to recognize and thank (in no specific order):

- **Dr. Nathaniel Frissell** at the University of Scranton — for his guidance, encouragement, and support throughout this project. His dedication to bridging citizen science and radio propagation research has been a constant source of inspiration.
- **Rob Suggs** at Central Connecticut State University and **Dr. Jay Weitzen** at the University of Massachusetts Lowell — for aid in providing technical expertise, guidance, code, and sample data.
- **Gary Mikitin** — for technical guidance, organization of the Meteor Scatter QSO Parties, and for sharing all results.
- **Dr. Mary Lou West** at Montclair State University and **Dr. Kuldeep Pandey** at the New Jersey Institute of Technology — for their technical expertise, support, and contributions to this work.
- **The broader HamSCI community** — whose passion for amateur radio science and willingness to collect and share data made this research possible.
- **All amateur radio citizen scientists** who participated in the Meteor Scatter QSO Parties and uploaded data submissions to Zenodo, making this dataset accessible for scientific study.
- The author made use of **Claude (Anthropic)** AI for assistance with code development and data visualization during this project.

---

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{tormann2026,
  author    = {Tormann, Nina},
  title     = {Analyzing Meteor Scatter Communications Through Citizen Science
               and Data-Driven Methods},
  school    = {University of Scranton},
  year      = {2026},
  type      = {Bachelor of Science Thesis},
  note      = {Department of Physics and Engineering. Supervisor: Dr. Nathaniel Frissell}
}
```
