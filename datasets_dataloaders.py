from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------
# 1. Creating a Dataset
# ---------------------------------------------------------------
# TensorDataset takes your input features (X) and labels (y)
# and combines them into a single dataset object.
#
# Why? Because PyTorch models expect input and target pairs,
# and TensorDataset makes it easy to keep them together.
#
# Think of it like a zip(X, y) that is compatible with PyTorch.
#
# Each item in this dataset will be:
#    dataset[i] → (X[i], y[i])
#
# Requirements:
# - X and y MUST be tensors
# - They must have the same first dimension (same number of samples)
#
# Example:
# If X shape = (100, 3)  → 100 samples, 3 features each
# and y shape = (100, 1) → 100 labels
# then dataset[i] returns (X[i], y[i])
# ---------------------------------------------------------------
dataset = TensorDataset(X, y)

# ---------------------------------------------------------------
# 2. Creating a DataLoader
# ---------------------------------------------------------------
# DataLoader automatically breaks the dataset into small batches.
# Why do we need batching?
# - Training on all data at once is slow and unstable
# - Batches give faster training and smoother optimization
#
# Parameters:
# - dataset: the TensorDataset created above
# - batch_size=16 → each batch will contain 16 samples
# - shuffle=True  → randomly shuffle the dataset each epoch
#
# This creates an iterator that returns:
#    batch_x → tensor of shape (16, number_of_features)
#    batch_y → tensor of shape (16, number_of_outputs)
#
# Example:
# If X is (100, 3):
#   First batch_x shape  = (16, 3)
#   First batch_y shape  = (16, 1)
#
# DataLoader is essential in training loops:
#    for batch_x, batch_y in loader:
#        ...train model...
# ---------------------------------------------------------------
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------------------------------------------------------------
# 3. Looping Through Batches
# ---------------------------------------------------------------
# This loop retrieves batches from the DataLoader.
#
# Each iteration gives one mini-batch:
# - batch_x → a batch of inputs
# - batch_y → corresponding labels
#
# Printing the shapes shows how the data is divided.
#
# The 'break' ends the loop after the first batch.
# ---------------------------------------------------------------
for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)
    break
