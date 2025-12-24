# 🔮 PowerPredict - AI Lottery Number Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0+-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**PowerPredict** is an advanced lottery number prediction system that uses **deep learning**, **statistical analysis**, and **ensemble methods** to generate Powerball and Mega Millions number predictions based on historical drawing data.

> ⚠️ **Disclaimer**: Lottery outcomes are random. This tool is for educational and entertainment purposes only. Please gamble responsibly.

---

## ✨ Features

- 🧠 **Deep Learning Ensemble** - Transformer + Bidirectional LSTM/GRU hybrid neural networks
- 📊 **Multi-Strategy Analysis** - Frequency, gap, Markov chain, and pattern matching models
- 🎯 **Smart Diversity** - Guaranteed unique predictions with Hamming distance enforcement
- 📈 **Historical Data Analysis** - Analyzes 1,800+ historical lottery drawings
- 🐳 **Docker Ready** - Containerized deployment with one command
- ⚡ **Fast Predictions** - Optimized numpy/pandas operations

---

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone and build
git clone https://github.com/cpeoples/powerpredict.git
cd powerpredict
docker build -t powerpredict .

# Run predictions
docker run powerpredict powerball -n 5
docker run powerpredict megamillions -n 10 --analyze
```

### Using Python

```bash
# Clone repository
git clone https://github.com/cpeoples/powerpredict.git
cd powerpredict

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run predictions
python main.py powerball -n 5
python main.py megamillions -n 10 --analyze
```

---

## 📖 Usage

```bash
# Generate 5 Powerball predictions
python main.py powerball -n 5

# Generate 10 Mega Millions predictions with statistical analysis
python main.py megamillions -n 10 --analyze

# Quick mode (statistical only, no deep learning)
python main.py powerball -n 5 --quick

# Show help
python main.py --help
```

### Command Line Options

| Option                  | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| `powerball`             | Predict Powerball numbers (1-69 + Power Ball 1-26)    |
| `megamillions`          | Predict Mega Millions numbers (1-70 + Mega Ball 1-25) |
| `-n, --num-predictions` | Number of predictions to generate (default: 5)        |
| `-a, --analyze`         | Show detailed statistical analysis                    |
| `-q, --quick`           | Skip deep learning (faster, statistical only)         |

---

## 🧪 How It Works

PowerPredict combines **four prediction strategies** into a master ensemble:

### 1. Weighted Statistical Model

Analyzes historical frequency, gap patterns, and positional tendencies to score each number.

### 2. Markov Chain Model

Uses transition probabilities to predict which numbers are likely to follow recent drawings.

### 3. Pattern Matching Model

Generates combinations matching historical patterns (sum ranges, odd/even ratios, high/low distribution).

### 4. Deep Learning Ensemble

- **Transformer** with multi-head attention for sequence patterns
- **Bidirectional LSTM/GRU** hybrid for temporal dependencies
- Temperature-based sampling to prevent mode collapse

### Master Ensemble

Combines all models with weighted averaging and enforces diversity:

- ✅ No repeated Power Balls across predictions
- ✅ Minimum Hamming distance between number sets
- ✅ Full range coverage (no clustering)

---

## 🛠️ Technology Stack

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" height="50"/>
  &nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" alt="TensorFlow" height="50"/>
  &nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg" alt="Keras" height="50"/>
  &nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" alt="Pandas" height="50"/>
  &nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-learn" height="50"/>
</p>

| Technology   | Version | Purpose               |
| ------------ | ------- | --------------------- |
| Python       | 3.11+   | Core runtime          |
| TensorFlow   | 2.18+   | Deep learning backend |
| Keras        | 3.0+    | Neural network API    |
| NumPy        | 1.26+   | Numerical computing   |
| Pandas       | 2.2+    | Data manipulation     |
| Scikit-learn | 1.5+    | ML utilities          |

---

## 📊 Sample Output

```
======================================================================
🔮 POWERPREDICT - INTELLIGENT LOTTERY ANALYSIS SYSTEM
======================================================================
   Game: POWERBALL
   Predictions: 5
   Mode: Full (Statistical + Deep Learning)

📥 Loading historical data...
   ✓ Loaded 1885 historical drawings

📊 Running comprehensive statistical analysis...
   ✓ Analysis complete

🧠 Training deep learning ensemble...
   ✓ Training complete

======================================================================
⭐ MASTER ENSEMBLE PREDICTIONS (HIGHEST CONFIDENCE)
======================================================================

🎰 MASTER ENSEMBLE:
--------------------------------------------------
   #1: [ 6 - 22 - 32 - 51 - 57]  Power Ball: 22 (agreement: 46%)
   #2: [ 6 - 19 - 39 - 53 - 66]  Power Ball: 12 (agreement: 44%)
   #3: [ 7 - 30 - 40 - 49 - 63]  Power Ball: 18 (agreement: 57%)
   #4: [ 9 - 22 - 42 - 49 - 58]  Power Ball: 15 (agreement: 54%)
   #5: [ 8 - 13 - 21 - 42 - 53]  Power Ball:  7 (agreement: 47%)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This software is provided for **educational and entertainment purposes only**. Lottery outcomes are determined by cryptographically secure random number generators and cannot be predicted by any statistical or machine learning method.

- The probability of winning Powerball is 1 in 292,201,338
- The probability of winning Mega Millions is 1 in 302,575,350

**Please gamble responsibly.**

---

<p align="center">
  Made with ❤️ and 🤖 by <a href="https://github.com/cpeoples">cpeoples</a>
</p>
