import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np

def main():
    """
    Parameters:
    systems (list): List of systems containing CSV datasets.
    num_repeats (int): Number of times to repeat the evaluation for avoiding stochastic bias.
    train_frac (float): Fraction of data to use for training.
    random_seed (int): Initial random seed to ensure the results are reproducible.
    """

    # Specify the parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3  # Modify this value to change the number of repetitions
    train_frac = 0.7  # Modify this value to change the training data fraction (e.g., 0.7 for 70%)
    random_seed = 1  # The random seed will be altered for each repeat

    # Initialize the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=random_seed)  # You can adjust 'n_estimators' or other hyperparameters

    for current_system in systems:
        datasets_location = 'datasets/{}'.format(current_system)  # Modify this to specify the location of the datasets

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]  # List all CSV files in the directory

        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repetitions: {num_repeats}')

            data = pd.read_csv(os.path.join(datasets_location, csv_file))  # Load data from CSV file

            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}  # Initialize a dict to store results for repeated evaluations

            for current_repeat in range(num_repeats):  # Repeat the process n times
                # Randomly split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=random_seed*current_repeat)  # Change the random seed based on the current repeat
                test_data = data.drop(train_data.index)

                # Split features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Train the model
                model.fit(training_X, training_Y)  # Train the Random Forest model with the training data

                # Predict on the test data
                predictions = model.predict(testing_X)

                # Calculate evaluation metrics for the current repeat
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                # Store the metrics
                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Calculate the average of the metrics for all repeats
            print(f'Average MAPE for Random Forest: {np.mean(metrics["MAPE"]):.2f}')
            print(f"Average MAE for Random Forest: {np.mean(metrics['MAE']):.2f}")
            print(f"Average RMSE for Random Forest: {np.mean(metrics['RMSE']):.2f}")

if __name__ == "__main__":
    main()
