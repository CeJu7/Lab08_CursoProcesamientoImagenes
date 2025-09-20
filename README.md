# DSP Repository - Virtual Environment Setup

This repository contains Digital Signal Processing (DSP) examples and exercises in Python.

## 🚀 Quick Start

### 1. Activate the Virtual Environment
```bash
source activate_env.sh
```

### 2. Alternative Manual Activation
```bash
source .venv/bin/activate
```

### 3. Deactivate Environment
```bash
deactivate
```

## 📦 Installed Dependencies

The virtual environment includes all necessary packages for DSP work:

- **numpy** (2.3.3) - Numerical operations and arrays
- **matplotlib** (3.10.6) - Plotting and visualization
- **scipy** (1.16.2) - Scientific computing and signal processing
- **sounddevice** (0.5.2) - Audio recording and playback
- **ipython** (9.5.0) - Enhanced Python shell
- **jupyter** (1.1.1) - Jupyter notebooks support

## 📁 Repository Contents

- `DSP_mel.py` - Mel frequency analysis and filtering
- `DSP_FFT_DCT.py` - FFT and DCT comparison with interactive demos
- `DSP_pre_enfasis.py` - Pre-emphasis filtering for audio
- `DSP_ventaneo.py` - Windowing functions and frame analysis
- `DSP_basics_examples.ipynb` - Basic DSP concepts notebook
- `DSP_lab_operaciones_senales_discretas.ipynb` - Discrete signal operations lab

## 🔧 Requirements

- Python 3.13+
- macOS (for sounddevice compatibility)

## 📋 Dependencies File

All dependencies are listed in `requirements.txt`. To install in a new environment:

```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. Activate the environment
2. Run Python scripts directly or start Jupyter:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open and run the notebooks or Python files

## ⚠️ Audio Dependencies

The `sounddevice` package requires:
- **macOS**: Core Audio (built-in)
- **Linux**: ALSA development files
- **Windows**: Windows audio drivers

## 🆘 Troubleshooting

- If sounddevice issues occur, ensure your system has proper audio drivers
- For notebook issues, try: `jupyter lab --generate-config`
- To reset environment: delete `.venv` folder and run setup again
