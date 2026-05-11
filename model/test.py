import numpy as np
import pandas as pd
import pickle
import os

Wc = np.random.randn(8, 1, 3, 3) * np.sqrt(2. / 9)
print(Wc.shape)
print(Wc)