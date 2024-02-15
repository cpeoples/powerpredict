#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define constants for game-specific information
GAMES = {
    "megamillions": {
        "ball": "Mega Ball",
        "featured_ball": "Megaplier",
        "game": "Mega_Millions",
        "featured_range": 25,
        "high_range": 70
    },
    "powerball": {
        "ball": "Power Ball",
        "featured_ball": "Power Play",
        "game": "Powerball",
        "featured_range": 26,
        "high_range": 69
    }
}


def load_dataset(game):
    try:
        # Retrieve all historical data in CSV file
        # Retrieve the last n lines in CSV file
        # data = pd.read_csv(
        #    f"https://www.texaslottery.com/export/sites/lottery/Games/{game}/Winning_Numbers/{sys.argv[1]}.csv", header=None).tail(100)

        # Retrieve all historical data in CSV file
        data = pd.read_csv(
            f"https://www.texaslottery.com/export/sites/lottery/Games/{GAMES[game]['game']}/Winning_Numbers/{game}.csv", header=None)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()

    data.columns = ["Game Name", "Month", "Day", "Year", "Num1",
                    "Num2", "Num3", "Num4", "Num5", GAMES[game]["ball"], GAMES[game]["featured_ball"]]
    return data


def preprocess_dataset(data, game):
    required_data = data[["Num1", "Num2", "Num3",
                          "Num4", "Num5", GAMES[game]["ball"]]]

    # Preprocess dataset: separate main numbers and featured ball number
    main_numbers = required_data[["Num1", "Num2", "Num3", "Num4", "Num5"]]
    ball_number = required_data[GAMES[game]["ball"]]

    # Scale the main numbers between 1 and high_range
    scaler_main = MinMaxScaler(feature_range=(1, GAMES[game]["high_range"]))
    scaled_main_numbers = scaler_main.fit_transform(main_numbers)

    # Scale the featured ball number between 1 and featured_range
    scaler_number = MinMaxScaler(
        feature_range=(1, GAMES[game]["featured_range"]))
    scaled_number = scaler_number.fit_transform(
        np.array(ball_number).reshape(-1, 1))

    # Combine the scaled main numbers and featured ball number
    scaled_data = np.column_stack((scaled_main_numbers, scaled_number))

    return scaled_data, scaler_main, scaler_number


def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Reshape to [samples, time_steps, n_features]
    train_data = np.reshape(
        train_data, (train_data.shape[0], 1, train_data.shape[1]))
    test_data = np.reshape(
        test_data, (test_data.shape[0], 1, test_data.shape[1]))

    return train_data, test_data


def create_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(50, input_shape=input_shape))
    model.add(keras.layers.Dense(input_shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def train_model(model, train_data):
    model.fit(train_data, train_data, epochs=20, batch_size=1, verbose=0)


def generate_prediction(model, test_data, scaler_main, scaler_number):
    prediction = model.predict(test_data)

    # Convert the scaled prediction back to the original range and round it to the nearest integer
    prediction = np.rint(np.column_stack((scaler_main.inverse_transform(
        prediction[:, :5]), scaler_number.inverse_transform(prediction[:, 5:]))))

    return prediction


def validate_prediction(prediction, game):
    for i in range(len(prediction)):
        unordered_prediction = prediction[i][:5]

        # Ensure all main numbers are unique and within the valid range by replacing invalid ones
        while len(set(unordered_prediction)) != len(unordered_prediction) or (unordered_prediction < 1).any() or (unordered_prediction > GAMES[game]["high_range"]).any():
            # Find the first invalid number (either duplicate or out of range)
            invalid = next(x for x in unordered_prediction if unordered_prediction.tolist(
            ).count(x) > 1 or x < 1 or x > GAMES[game]["high_range"])
            # Replace it with a new random number within the valid range
            unordered_prediction[unordered_prediction.tolist().index(
                invalid)] = np.random.randint(1, GAMES[game]["high_range"] + 1)

        # Make sure the featured ball number is within the valid range
        feature_ball = prediction[i][5]
        while feature_ball < 1 or feature_ball > GAMES[game]["featured_range"]:
            # Generate a new random number within the valid range
            feature_ball = np.random.randint(
                1, GAMES[game]["featured_range"] + 1)

        # Append the featured ball number (without sorting)
        final_prediction = np.append(unordered_prediction, feature_ball)
        print(
            f"Predicted {game.capitalize()} Draw {i+1}: {final_prediction}".replace(".", ""))


def main():
    # Check if the command-line argument is valid
    if len(sys.argv) < 2 or sys.argv[1] not in GAMES:
        print("Invalid argument. Please provide either 'megamillions' or 'powerball'.")
        sys.exit()

    game = sys.argv[1]
    data = load_dataset(game)
    scaled_data, scaler_main, scaler_number = preprocess_dataset(data, game)
    train_data, test_data = split_data(scaled_data)
    model = create_model((train_data.shape[1], train_data.shape[2]))
    train_model(model, train_data)
    prediction = generate_prediction(
        model, test_data, scaler_main, scaler_number)
    validate_prediction(prediction, game)


if __name__ == "__main__":
    main()
