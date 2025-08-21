# 🚀 Machine Learning & Deep Learning Template

A comprehensive template to start Machine Learning and Deep Learning projects with complete setup from library imports to model deployment.

## 📋 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [Template Structure](#-template-structure)
- [Installation Guide](#-installation-guide)
- [Use Cases](#-use-cases)
- [Best Practices](#-best-practices)
- [Troubleshooting](#-troubleshooting)

## 🎯 Overview

This template provides:
- ✅ Complete import statements for all popular ML/DL libraries
- ✅ Comprehensive requirements.txt with explanation for each library
- ✅ Optimal environment setup for reproducibility
- ✅ Easy-to-customize modular structure
- ✅ Usage guides for various cases

## 🚀 Quick Start

### 1. Clone/Download Template

```bash
git clone https://github.com/AkmaLeyzal/MachineLearning-Template.git
cd MachineLearning-Template
```

### 2. Setup Virtual Environment
#### For Linux/Mac using Venv
```bash
python -m venv ml_env       # you can change "ml_env" as you want
source ml_env/bin/activate  
```

#### For Windows using Venv
```bash
python -m venv ml_env       # you can change "ml_env" as you want
ml_env\Scripts\activate     
```

#### using Conda
```bash
conda create -n ml_env python=3.9   # you can change "ml_env" as you want
conda activate ml_env
```

### 3. Install Dependencies
#### Install according to use case (see requirements.txt)

```bash
pip install -r requirements_minimal.txt
```

### 4. Start Jupyter Notebook or Another Notebook

```bash
jupyter lab
```

### 5. Customize Template

- Open the main template file
- Uncomment required libraries
- Start coding! 😝😝😝

## 📁 Template Structure

```
MachineLearning-Template/
├── Notebooks/
│   └── MachineLearning-Template      # Main and Fully completed template
├── Datasets/
│   ├── RawData/                      
│   ├── ProcessedData/                
│   └── ExternalData/                  
├── Models/
│   ├── DummyModel/                  
│   ├── HyperparameterModel/
│   ├── TestingModel/                    
│   └── FinalModel/                        
├── requirements
├── LICENSE
└── README.md
```

## 🛠 Installation Guide

### Install library according to use case
- Open requirements.txt
- Uncomment using `Ctrl + /` or Delete `#` for library you will use
- Then Run `pip install -r requirements.txt` on your terminal


### Platform-Specific Notes

#### 🍎 Apple Silicon (M-chip series) Macs
```bash
# For TensorFlow
pip install tensorflow-macos tensorflow-metal

# For PyTorch
pip install torch torchvision
```

#### 🐧 Linux with GPU
```bash
# NVIDIA GPU support
pip install tensorflow[and-cuda]
# or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Use Cases

### 1. **Classification Project**
```python
# Uncomment di template:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
```

### 2. **Deep Learning Image Classification**
```python
# Uncomment di template:
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

### 3. **NLP Sentiment Analysis**
```python
# Uncomment di template:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
```

### 4. **Time Series Forecasting**
```python
# Uncomment di template:
from prophet import Prophet
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
```

## ✅ Best Practices

### 1. **Environment Management**
- Always use virtual environment
- Pin exact versions for production (`==` instead of `>=`)
- Document actually used dependencies

### 2. **Import Management**
- Only uncomment libraries that are actually needed
- Group imports according to functionality
- Use lazy imports for large libraries

### 3. **Reproducibility**
- Set random seeds (already in template)
- Document used versions
- Use fixed train/validation splits

### 4. **GPU Optimization**
```python
# TensorFlow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# PyTorch
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. **Import Error: Module not found**
```bash
# Solution: Install missing module
pip install [module-name]

# Check installed packages
pip list
```

#### 2. **CUDA/GPU Issues**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### 3. **SSL Certificate Issues**
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org [package]
```

## 📚 Additional Resources

### Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Hugging Face Course](https://huggingface.co/course)

### Useful Tools
- [Jupyter Notebook Extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)
- [MLflow for Experiment Tracking](https://mlflow.org/)
- [Weights & Biases](https://wandb.ai/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

### Datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [Papers with Code Datasets](https://paperswithcode.com/datasets)

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new useful libraries
- Improve documentation
- Fix bugs or issues
- Add new template variations

## 📄 License

This template is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Scikit-learn team for excellent ML library
- TensorFlow and PyTorch teams for deep learning frameworks
- Hugging Face for transformers library
- All open-source contributors in the ML/AI community

---

**Happy Machine Learning! 🚀**

*This template will be continuously updated along with the evolution of the ML/DL ecosystem*

© 2023 Akmaleyzal