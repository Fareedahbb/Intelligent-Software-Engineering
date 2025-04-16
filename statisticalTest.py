import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel

def main():
    """
    Runs both Random Forest and Linear Regression across multiple systems,
    prints average MAPE, MAE, RMSE for each, and computes a paired t-test
    on their average MAEs.
    """

    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3
    train_frac = 0.7
    random_seed = 1

    # Collect per-dataset average MAE for statistical test
    rf_mae_list = []
    lr_mae_list = []

    for current_system in systems:
        datasets_location = f'datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repetitions: {num_repeats}')
            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            # Random Forest evaluation
            rf_model = RandomForestRegressor(n_estimators=100, random_state=random_seed)
            rf_metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}
            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed*current_repeat)
                test_data = data.drop(train_data.index)
                X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
                X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

                rf_model.fit(X_train, y_train)
                preds_rf = rf_model.predict(X_test)
                rf_metrics['MAPE'].append(mean_absolute_percentage_error(y_test, preds_rf))
                rf_metrics['MAE'].append(mean_absolute_error(y_test, preds_rf))
                rf_metrics['RMSE'].append(np.sqrt(mean_squared_error(y_test, preds_rf)))

            avg_rf_mape = np.mean(rf_metrics['MAPE'])
            avg_rf_mae  = np.mean(rf_metrics['MAE'])
            avg_rf_rmse = np.mean(rf_metrics['RMSE'])
            print(f'Average MAPE for Random Forest: {avg_rf_mape:.2f}')
            print(f'Average MAE for Random Forest:  {avg_rf_mae:.2f}')
            print(f'Average RMSE for Random Forest: {avg_rf_rmse:.2f}')
            rf_mae_list.append(avg_rf_mae)

            # Linear Regression baseline
            lr_model = LinearRegression()
            lr_metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}
            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed*current_repeat)
                test_data = data.drop(train_data.index)
                X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
                X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

                lr_model.fit(X_train, y_train)
                preds_lr = lr_model.predict(X_test)
                lr_metrics['MAPE'].append(mean_absolute_percentage_error(y_test, preds_lr))
                lr_metrics['MAE'].append(mean_absolute_error(y_test, preds_lr))
                lr_metrics['RMSE'].append(np.sqrt(mean_squared_error(y_test, preds_lr)))

            avg_lr_mape = np.mean(lr_metrics['MAPE'])
            avg_lr_mae  = np.mean(lr_metrics['MAE'])
            avg_lr_rmse = np.mean(lr_metrics['RMSE'])
            print(f'Average MAPE for Linear Regression: {avg_lr_mape:.2f}')
            print(f'Average MAE for Linear Regression:  {avg_lr_mae:.2f}')
            print(f'Average RMSE for Linear Regression: {avg_lr_rmse:.2f}')
            lr_mae_list.append(avg_lr_mae)

    # Paired t-test on average MAEs across datasets
    t_stat, p_value = ttest_rel(lr_mae_list, rf_mae_list)
    print("\n--- Paired t-test on Average MAE (LR vs RF) ---")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("The difference in MAE between Linear Regression and Random Forest is statistically significant.")
    else:
        print("No statistically significant difference in MAE between Linear Regression and Random Forest.")

if __name__ == "__main__":
    main()

