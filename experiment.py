import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
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

    systems = ['batlik','dconvert','h2','jump3r','kanzi','lrzip','x264','xz','z3']
    num_repeats = 3
    train_frac = 0.7
    random_seed = 1

    # we'll store one row per (system, dataset, model)
    all_results = []

    for current_system in systems:
        datasets_location = f'datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}')

            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            # we'll compare four models
            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(random_state=random_seed),
                'DecisionTree': DecisionTreeRegressor(random_state=random_seed),
                'SVM': SVR()
            }

            for model_name, model in models.items():
                metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

                for i in range(num_repeats):
                    # split
                    train = data.sample(frac=train_frac, random_state=random_seed*i)
                    test  = data.drop(train.index)

                    X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]
                    X_test,  y_test  = test.iloc[:,:-1],  test.iloc[:,-1]

                    # train & predict
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # compute errors
                    metrics['MAPE'].append(mean_absolute_percentage_error(y_test, preds))
                    metrics['MAE'].append(mean_absolute_error(y_test, preds))
                    metrics['RMSE'].append(np.sqrt(mean_squared_error(y_test, preds)))

                # compute averages
                avg_mape = np.mean(metrics['MAPE'])
                avg_mae  = np.mean(metrics['MAE'])
                avg_rmse = np.mean(metrics['RMSE'])

                # print as before
                print(f"  {model_name:16} Average MAPE: {avg_mape:.2f}, MAE: {avg_mae:.2f}, RMSE: {avg_rmse:.2f}")

                # collect for CSV
                all_results.append({
                    'System': current_system,
                    'Dataset': csv_file,
                    'Model': model_name,
                    'Avg_MAPE': avg_mape,
                    'Avg_MAE': avg_mae,
                    'Avg_RMSE': avg_rmse
                })

    # at the end: dump to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\n All results saved to model_comparison_results.csv")

if __name__ == "__main__":
    main()
