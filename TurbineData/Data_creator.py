import pandas as pd

# --- 1. Load the Data ---
# Replace the file path with your actual path to FD_004_train.txt
data = pd.read_csv("/Users/Garry/Desktop/HackaFuture/TurbineData/RUL_FD001.txt", sep=" ", header=None)
print(data)
data=data.iloc[:, :-1]
print(data)

data.columns = ["RUL"]
data.to_csv("rul_001_train_processed.csv", index=False)

print(data)
print("Data processing complete. Processed file saved as 'FD_004_train_processed.csv'.")
