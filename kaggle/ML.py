import pandas as pd

housing_data_path = "./sample_submission.csv"
housing_data = pd.read_csv(housing_data_path)
housing_data.describe()

print("Welcome to VS Code Python !")