import optuna
import src.base_pipeline as bp
import src.GD_bias as gd
import src.dataloader as dl
import numpy as np
from joblib import Parallel, delayed

def objective(trial):
        rank = trial.suggest_int("rank", 1, 100)
        lambda_I = trial.suggest_float("lambda_I", 1e-5, 1e-1)
        mu_U = trial.suggest_float("mu_U", 1e-5, 1e-1)
        iter_n = trial.suggest_int("iter_n", 100, 1000)
        beta = trial.suggest_float("beta", 0.9, 0.999)
        
        model = gd.GDWithBias(rank, lambda_I, mu_U, iter_n, beta, verbose=False)
        model.fit(train_matrix)
        
        return model.score_hist[-1, 1]
    
def optimize(n_trials, name):
    study = optuna.load_study(study_name=name, storage='sqlite:///' + str(name) + '.db')
    study.optimize(objective, n_trials=n_trials)
    

if __name__ == "__main__":
    n_trials = 100
    
    dataloader = dl.Dataloader(train_ratio=0.80)
    train_matrix, test_matrix = dataloader.load_data()
    
    # Print some stats
    print("Train matrix non-zero count:", np.count_nonzero(train_matrix))
    print("Test matrix non-zero count:", np.count_nonzero(test_matrix))
    
    # Optimize the hyperparameters
    name = "Accuracy simple rounding" 
    study = optuna.create_study(direction="maximize", study_name=name, storage='sqlite:///' + str(name) + '.db', load_if_exists = True)

    r = Parallel(n_jobs=-1)([delayed(optimize)(1, name) for _ in range(n_trials)])
    
    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    model = gd.GDWithBias(**study.best_params, missing_values=True, verbose=True, y_true=test_matrix)
    
    model.plot_loss()