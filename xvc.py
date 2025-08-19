
import pandas as pd
import numpy as np
import torch

data = pd.read_csv('/Users/Garry/Desktop/HackaFuture/pump_sensor.csv')
print(data['machine_status'].unique())