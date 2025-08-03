import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

coffee = pd.read_csv('starbucks_customers.csv')
spent = coffee['spent']

spent_reshaped = np.array(spent).reshape(-1, 1)

mmscaler = MinMaxScaler()

reshaped_scaled = mmscaler.fit_transform(spent_reshaped)

print(np.min(reshaped_scaled))
print(np.max(reshaped_scaled))