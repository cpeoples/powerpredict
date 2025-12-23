# PowerPredict

PowerPredict employs TensorFlow to implement a Long Short-Term Memory (LSTM) network for the purpose of generating Powerball and Megamillions lottery numbers. The LSTM, a variant of Recurrent Neural Network (RNN), possesses the ability to learn and predict by considering long-term dependencies. This characteristic renders it potentially well-suited for time series prediction.

## Technology

<center><div style="display: inline_block"><br/>
<img align="center" alt="python" height="50" width="50" src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" />
<img align="center" alt="tensorflow" height="50" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg" />
<img align="center" alt="keras" height="50" src="https://keras.io/img/logo-small.png" />
</center>

<br/>

## How to Install and Run Locally

#### Python Installation

```bash
sudo apt update
sudo apt install python3 python3-pip
```

#### Alias

```bash
alias python=python3
alias pip=pip3
```

#### Setup Virtual Environment and Install Dependencies

```bash
git clone git@github.com:cpeoples/powerpredict.git
cd powerpredict
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

#### Execute

```bash
python main.py megamillions
python main.py powerball
```

The model will print predicted Powerball or Megamillions numbers.

<br/>

# Disclaimer

This particular model is solely for experimental purposes and should not be relied upon as a dependable or precise means of forecasting Powerball or Megamillions numbers. The outcome of the lottery is determined through a random process that cannot be accurately foreseen by this or any other statistical approach. Any utilization of this model is done at your own risk, and it should not be utilized as a foundation for engaging in any type of gambling activities.
