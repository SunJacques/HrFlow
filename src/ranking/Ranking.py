import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import base_pipeline as bp

class Ranking(bp.RankingMethod):
    def __init__(self, interaction_matrix, job_embeddings):
        """
        Initialize Ranking model with interaction matrix and job embeddings.
        :param interaction_matrix: Sparse matrix (users x jobs) with interaction scores (1 for view, 5 for apply).
        :param job_embeddings: Dictionary mapping job IDs to embedding vectors.
        """
        self.interaction_matrix = interaction_matrix
        self.job_embeddings = job_embeddings
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_training_data(self):
        """
        Generate training data from the entire interaction matrix.
        :return: Feature matrix X and labels y.
        """
        features = []
        labels = []
        
        num_users, num_jobs = self.interaction_matrix.shape
        
        for user_id in range(num_users):
            user_interactions = self.interaction_matrix[user_id].toarray().flatten()
            
            for job_id in range(num_jobs):
                score = user_interactions[job_id] if job_id < len(user_interactions) else 0
                # job_embedding = self.job_embeddings.get(job_id, np.zeros(300))  # Assuming 300-dim embeddings
                
                # Generate features (interaction score + job embedding)
                # features.append(np.concatenate(([score], job_embedding)))
                features.append(score)
                labels.append(score)
        
        # Normalize features
        X = self.scaler.fit_transform(features)
        y = np.array(labels)
        return X, y
    
    def train(self):
        """
        Train the ranking model using the entire dataset.
        """
        X, y = self.prepare_training_data()
        train_data = lgb.Dataset(X, label=y)
        params = {'objective': 'lambdarank', 'metric': 'ndcg', 'verbosity': -1}
        self.model = lgb.train(params, train_data, num_boost_round=50)
    
    def top_k(self, user_id, k=10):
        """
        Rank the Top-K jobs for a given user using the trained model.
        :param user_id: ID of the user.
        :param k: Number of top jobs to return.
        :return: List of Top-K ranked job IDs.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        
        user_interactions = self.interaction_matrix[user_id].toarray().flatten()
        features = []
        job_ids = []
        
        for job_id in range(len(user_interactions)):
            score = user_interactions[job_id]
            job_embedding = self.job_embeddings.get(job_id, np.zeros(300))
            features.append(np.concatenate(([score], job_embedding)))
            job_ids.append(job_id)
        
        X = self.scaler.transform(features)
        scores = self.model.predict(X)
        ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order
        
        return [job_ids[i] for i in ranked_indices[:k]]
    
    def predict(self, start_user_id=15882, action="apply", output_file="predictions.csv"):
        """
        Generate predictions for all users starting from row 15882 of the interaction matrix
        and save the results in a CSV file.
        :param start_user_id: Starting user ID for predictions.
        :param action: Action type (default: "apply").
        :param output_file: Output CSV file name.
        """
        predictions = []
        num_users, _ = self.interaction_matrix.shape
        
        for user_id in range(start_user_id, num_users):
            top_10_jobs = self.top_k(user_id, k=10)
            predictions.append([user_id - start_user_id, action, top_10_jobs])
        
        df = pd.DataFrame(predictions, columns=["session_id", "action", "job_id"])
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
            
            
