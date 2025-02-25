import numpy as np
import scipy.sparse as sp
import pickle

class Dataloader:
    def __init__(self, train_ratio=0.80):
        with open("data/sparse_matrix.pkl", "rb") as f:
            self.data = pickle.load(f).toarray()
        
        self.train_ratio = train_ratio
        self.test_data_index = 15882

    def load_data(self):
        non_zero_positions = np.argwhere(self.data != 0)
        non_zero_values = self.data[self.data != 0]
        
        # Shuffle positions and values together
        shuffle_indices = np.random.permutation(len(non_zero_values))
        non_zero_positions = non_zero_positions[shuffle_indices]
        non_zero_values = non_zero_values[shuffle_indices]
        
        # Split into train and test
        num_to_add = int(len(non_zero_values) * self.train_ratio)
        
        train_matrix = np.zeros_like(self.data)
        test_matrix = np.zeros_like(self.data)
        
        for (i, j), value in zip(non_zero_positions[:num_to_add], non_zero_values[:num_to_add]):
            train_matrix[i, j] = value
        
        for (i, j), value in zip(non_zero_positions[num_to_add:], non_zero_values[num_to_add:]):
            test_matrix[i, j] = value
        
        return train_matrix, test_matrix
    
    def load_test(self):
        return self.data[self.test_data_index:]

# Load and split the data
dataloader = Dataloader()
train_matrix, test_matrix = dataloader.load_data()

# Print some stats
print("Train matrix non-zero count:", np.count_nonzero(train_matrix))
print("Test matrix non-zero count:", np.count_nonzero(test_matrix))
