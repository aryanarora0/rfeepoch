import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from crfe import custom_rfe

x = torch.randn(100, 10)
y = torch.randn(100)

desired_features = 5

selected_features, elimination_order = custom_rfe(x, y, desired_features)

#print(x, y)
print("Selected features after RFE:", selected_features)
print("Eliminated features:", elimination_order)