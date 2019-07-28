import pandas as pd

housing_data_path = "./data/sample_submission.csv"
housing_data = pd.read_csv(housing_data_path)
housing_data.columns

housing_data = housing_data.dropna()
y = housing_data.pr
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)

print("Welcome to VS Code Python !")