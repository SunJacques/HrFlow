import dataloader as dl
import matplotlib.pyplot as plt
import base_pipeline as bp

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

class SGDWithBias(bp.CollaborativeMethod):
    def __init__(self, rank, lambda_I, mu_U, iter_n, beta, batch_size, lr_I=1e-3, lr_U=1e-3, lr_B_U=1e-3, lr_B_I=1e-3, verbose=False):
        self.param = {
            'rank': rank,
            'lambda_I': lambda_I,
            'mu_U': mu_U,
            'iter_n': iter_n,
            'lr_I': lr_I,
            'lr_U': lr_U,
            'lr_B_U': lr_B_U,  # Learning rate for user bias
            'lr_B_I': lr_B_I,  # Learning rate for item bias
            'beta': beta,
            'batch_size': batch_size
        }
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.score_hist = np.array([]).reshape(0, 2)
        self.loss_hist = []
        self.bias_U = None
        self.bias_I = None
    
    def setup_optim(self, incomplete_matrix):
        self.I = torch.rand((incomplete_matrix.shape[0], self.param['rank']), requires_grad=True, device=self.device)
        self.U = torch.rand((incomplete_matrix.shape[1], self.param['rank']), requires_grad=True, device=self.device)
        self.bias_U = torch.zeros(incomplete_matrix.shape[1], requires_grad=True, device=self.device)  # User bias
        self.bias_I = torch.zeros(incomplete_matrix.shape[0], requires_grad=True, device=self.device)  # Item bias
        
        optimizer = torch.optim.SGD([
            {'params': self.I, 'lr': self.param["lr_I"]},
            {'params': self.U, 'lr': self.param["lr_U"]},
            {'params': self.bias_U, 'lr': self.param["lr_B_U"]},
            {'params': self.bias_I, 'lr': self.param["lr_B_I"]}
        ])
        return optimizer
    
    def fit(self, incomplete_matrix, test_matrix=None, save_model=False):
        incomplete_matrix = torch.tensor(incomplete_matrix, dtype=torch.float32, device=self.device)
        mask = (incomplete_matrix > 0).detach().cpu().numpy()
        
        # Create a DataLoader for mini-batch processing
        rows, cols = np.where(mask)
        dataset = TensorDataset(torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long), incomplete_matrix[rows, cols])
        dataloader = DataLoader(dataset, batch_size=self.param['batch_size'], shuffle=True)
        
        optimizer = self.setup_optim(incomplete_matrix)
        
        for i in tqdm(range(self.param['iter_n'])):
            epoch_loss = 0.0  # Accumulate loss over the epoch
            num_batches = 0
            
            for batch_rows, batch_cols, batch_ratings in dataloader:
                optimizer.zero_grad()
                
                # Predict ratings for the current batch
                batch_I = self.I[batch_rows]
                batch_U = self.U[batch_cols]
                batch_bias_U = self.bias_U[batch_cols]
                batch_bias_I = self.bias_I[batch_rows]
                batch_preds = (batch_I * batch_U).sum(dim=1) + batch_bias_U + batch_bias_I
                
                # Compute loss for the current batch
                batch_loss = self.masked_loss_bias(batch_preds, batch_ratings, batch_rows, batch_cols)
                
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()  # Accumulate batch loss
                num_batches += 1
            
            # Record the average loss for the epoch
            epoch_loss /= num_batches
            self.loss_hist.append(epoch_loss)
            
            if self.verbose:
                complete_matrix = self.predict()
                train_score = np.sqrt(mean_squared_error(incomplete_matrix[mask].detach().cpu().numpy(), complete_matrix[mask]))
                test_score = np.sqrt(mean_squared_error(test_matrix[test_matrix > 0], complete_matrix[test_matrix > 0]))
                self.score_hist = np.append(self.score_hist, [[train_score, test_score]], axis=0)
                print(f"Iteration {i+1}: Train RMSE = {train_score}, Test RMSE = {test_score}")
                print(f"Loss = {epoch_loss}")
                
        if save_model:
            torch.save({'I': self.I, 'U': self.U, 'bias_U': self.bias_U, 'bias_I': self.bias_I}, 'model_checkpoint.pth')
            print("Model saved successfully!")
    
    def masked_loss_bias(self, preds, targets, rows, cols):
        main = ((preds - targets) ** 2).sum()
        reg_I = self.param['lambda_I'] * (self.I[rows] ** 2).sum()
        reg_U = self.param['mu_U'] * (self.U[cols] ** 2).sum()
        reg_bias_U = self.param['beta'] * (self.bias_U[cols] ** 2).sum()
        reg_bias_I = self.param['beta'] * (self.bias_I[rows] ** 2).sum()
        return main + reg_I + reg_U + reg_bias_U + reg_bias_I
    
    def predict(self):
        bias_term = self.bias_U.unsqueeze(0) + self.bias_I.unsqueeze(1)  # Adding user and item biases
        return (self.I @ self.U.T + bias_term).detach().cpu().numpy()

    def plot_loss_history(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_hist, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()
        plt.savefig('loss_history.png')
        
        print(f"Final loss: {self.loss_hist[-1]}")
    
    def plot_score_history(self):
        if self.score_hist.shape[0] > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(self.score_hist[:, 0], label='Train RMSE')
            plt.plot(self.score_hist[:, 1], label='Test RMSE')
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('Score History')
            plt.legend()
            plt.savefig('score_history.png')
            
            print(f"Final train RMSE: {self.score_hist[-1, 0]}")
            print(f"Final test RMSE: {self.score_hist[-1, 1]}")
        else:
            print("Score history is empty. Enable verbose mode to track scores.")
        


if __name__ == "__main__":
    dataloader = dl.Dataloader(train_ratio=0.80)
    train_matrix, test_matrix = dataloader.load_data()
    
    model = SGDWithBias(100, 1e-3, 1e-3, 20, 0.1, batch_size=64 ,verbose=True)
    model.fit(train_matrix, test_matrix, save_model=True)
    
    model.plot_loss_history()
    model.plot_score_history()
    
    predictions = model.predict()
    n_high_score = predictions[predictions > 2.5].sum()
    n_low_score = predictions[predictions <= 2.5].sum()
    plt.figure(figsize=(8, 5))
    plt.hist(predictions, bins=20)
    plt.savefig('prediction_hist.png')
    print(f"Number of low score predictions: {n_low_score} | Number of high score predictions: {n_high_score}")