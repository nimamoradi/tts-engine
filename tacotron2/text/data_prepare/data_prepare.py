from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = pd.read_csv(filepath_or_buffer='metadata.csv', header=None, sep="|", dtype=np.str)
print(data.head(4))

train, other = train_test_split(data, test_size=0.15)
test, val = train_test_split(other, test_size=0.5)

train.to_csv('train.csv', header=None, sep='|', index=False)
test.to_csv('test.csv', header=None, sep='|', index=False)
val.to_csv('val.csv', header=None, sep='|', index=False)
