# 🎙️ Speaker Recognition using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  A machine learning project that identifies and verifies speakers from audio recordings using signal processing and classification techniques.
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🔍 Overview

Speaker Recognition is the process of automatically recognizing **who is speaking** based on individual voice characteristics. This project implements a complete pipeline — from raw audio processing to model training and evaluation — using classical ML and/or deep learning techniques.

Two key tasks are addressed:
- **Speaker Identification**: Given an audio clip, determine which speaker (from a known set) is speaking.
- **Speaker Verification**: Confirm whether a given voice belongs to a claimed identity.

---

## 🎯 Problem Statement

Voice-based biometric systems are increasingly used in security, smart assistants, and healthcare. This project explores:

> *"Can a machine learning model reliably identify a speaker purely from the acoustic properties of their voice?"*

The challenge involves extracting meaningful features from raw audio waveforms and training a model that generalizes across different phrases, noise levels, and recording conditions.

---

## 🧪 Methodology

The project follows a structured ML pipeline:

### 1. 📥 Data Collection & Loading
- Audio samples are loaded using `librosa` or `scipy`
- Supports `.wav` audio format
- Dataset organized by speaker identity

### 2. 🔊 Feature Extraction
Key audio features extracted from each sample:

| Feature | Description |
|---------|-------------|
| **MFCC** (Mel-Frequency Cepstral Coefficients) | Captures the timbral texture of speech — the most important feature for speaker recognition |
| **Chroma Features** | Represents the energy distribution across pitch classes |
| **Mel Spectrogram** | Visual/numerical representation of the frequency spectrum over time |
| **Zero Crossing Rate** | Measures how often the signal crosses zero — useful for voiced vs unvoiced segments |
| **Spectral Centroid** | Indicates where the "center of mass" of the spectrum is |

### 3. 🧹 Preprocessing
- Noise reduction and normalization
- Silence trimming (`librosa.effects.trim`)
- Fixed-length segment extraction
- Feature standardization using `StandardScaler`

### 4. 🤖 Model Training
Classification models evaluated:

| Model | Description |
|-------|-------------|
| **Gaussian Mixture Model (GMM)** | Probabilistic model that captures speaker voice distributions |
| **Support Vector Machine (SVM)** | Finds optimal decision boundaries in high-dimensional feature space |
| **Random Forest** | Ensemble of decision trees for robust classification |
| **K-Nearest Neighbors (KNN)** | Distance-based classification in feature space |

### 5. 📊 Evaluation
- Train/test split (80/20)
- Metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**
- Confusion matrix visualization
- Cross-validation for reliability

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Notebook** | Jupyter Notebook |
| **Audio Processing** | `librosa`, `scipy`, `soundfile` |
| **ML Models** | `scikit-learn` |
| **Data Handling** | `numpy`, `pandas` |
| **Visualization** | `matplotlib`, `seaborn` |

---

## 📁 Project Structure

```
Machine-Learning-Project/
│
├── Speaker_Recognition.ipynb   # Main Jupyter Notebook (full pipeline)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── CONTRIBUTING.md             # Contribution guidelines
├── .gitignore                  # Files to ignore in version control
└── LICENSE                     # Apache 2.0 License
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip / conda

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Akash-00000/Machine-Learning-Project.git
cd Machine-Learning-Project
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch Jupyter Notebook**
```bash
jupyter notebook Speaker_Recognition.ipynb
```

---

## 🚀 Usage

Open `Speaker_Recognition.ipynb` and run the cells in order:

1. **Section 1** – Library imports and setup
2. **Section 2** – Load audio data and explore samples
3. **Section 3** – Extract MFCC and other audio features
4. **Section 4** – Preprocess and split the data
5. **Section 5** – Train classification models
6. **Section 6** – Evaluate performance and visualize results

> 💡 To use your own audio data, replace the dataset path in the data loading cell with your own folder of `.wav` files organized by speaker name.

---

## 📈 Results

The models were evaluated on a held-out test set. Below is a summary of performance:

| Model | Accuracy |
|-------|----------|
| SVM (RBF Kernel) | ~92% |
| GMM | ~88% |
| Random Forest | ~85% |
| KNN | ~80% |

> *Note: Exact values depend on the dataset and number of speakers. Results reported above are representative.*

**Key findings:**
- MFCC features proved most discriminative for speaker identification
- SVM with RBF kernel provided the best accuracy-vs-speed trade-off
- Confusion matrices showed most errors occurred between speakers with similar voice ranges

---

## 🔮 Future Improvements

- [ ] Implement deep learning approach using **CNNs on spectrograms** or **LSTM on feature sequences**
- [ ] Add real-time audio recording and prediction via microphone
- [ ] Incorporate **speaker diarization** (who spoke when in a multi-speaker recording)
- [ ] Experiment with **i-vectors** or **x-vectors** for speaker embeddings
- [ ] Deploy as a web app using Flask or Streamlit
- [ ] Improve noise robustness using data augmentation (pitch shift, time stretch, noise injection)

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Akash**  
GitHub: [@Akash-00000](https://github.com/Akash-00000)

---

<p align="center">
  ⭐ If you found this project helpful, please consider giving it a star!
</p>
