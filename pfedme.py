import numpy as np
import copy
import matplotlib.pyplot as plt
import os

class Config:
    def __init__(self, num_clients=20, num_global_rounds=50, num_local_rounds=5, 
                 start_batch_size=20, lambda_val=15.0, lr=0.05, beta=1.0, 
                 k_inner_steps=5, dimension=10, data_per_client=30):
        self.num_clients = num_clients
        self.num_global_rounds = num_global_rounds
        self.num_local_rounds = num_local_rounds
        self.batch_size = start_batch_size
        self.lambda_val = lambda_val  # Lambda for Moreau Envelope regularization
        self.lr = lr  # Learning rate (eta)
        self.beta = beta  # Beta for global averaging
        self.k_inner_steps = k_inner_steps # Number of gradient descent steps to solve inner problem
        self.dimension = dimension
        self.data_per_client = data_per_client

class Client:
    def __init__(self, id, config, X, y, task_type="high_convexity"):
        self.id = id
        self.config = config
        self.X = X
        self.y = y
        self.task_type = task_type
        self.w_local = np.zeros(config.dimension)  # Local model parameter w_{i,r}
        self.theta = np.zeros(config.dimension)    # Personalized parameter theta_i
        
    def loss_function(self, theta, batch_idx):
        # f_i(theta)
        X_batch = self.X[batch_idx]
        y_batch = self.y[batch_idx]
        
        if self.task_type == "high_convexity" or self.task_type == "low_convexity":
            # Linear Regression / Quadratic Objective: 0.5 * ||X*theta - y||^2
            predictions = X_batch @ theta
            residuals = predictions - y_batch
            return 0.5 * np.mean(residuals**2) 
        
        return 0

    def gradient_function(self, theta, batch_idx):
        # \nabla f_i(theta)
        X_batch = self.X[batch_idx]
        y_batch = self.y[batch_idx]
        
        if self.task_type == "high_convexity" or self.task_type == "low_convexity":
            # Grad: X^T (X*theta - y) / N
            predictions = X_batch @ theta
            residuals = predictions - y_batch
            return (X_batch.T @ residuals) / len(y_batch)
            
        return np.zeros_like(theta)

    def find_personalized_theta(self, w_ref):
        """
        Solves the inner problem to find approx theta_i.
        min L(theta) + lambda/2 ||theta - w_ref||^2
        """
        theta_curr = np.copy(self.w_local) # Warm start with local model
        
        # Use full batch for demo stability
        indices = np.arange(len(self.y)) 
        
        for k in range(self.config.k_inner_steps):
            grad_f = self.gradient_function(theta_curr, indices)
            # Gradient of the proximal term: lambda * (theta - w_ref)
            grad_prox = self.config.lambda_val * (theta_curr - w_ref)
            
            # Update theta
            # Optimization step for inner problem
            theta_curr = theta_curr - self.config.lr * (grad_f + grad_prox)
            
        return theta_curr

    def local_training(self, w_global):
        """
        Performs R rounds of local updates.
        """
        self.w_local = np.copy(w_global) # Initialize w_{i,0} with w_global
        
        for r in range(self.config.num_local_rounds):
            # Step 1: Find Personalized Model \theta
            # argmin { f_i(theta) + lambda/2 ||theta - w_{i,r}||^2 }
            self.theta = self.find_personalized_theta(self.w_local)
            
            # Step 2: Update Local Model w (pFedMe update rule)
            # w_{i, r+1} = w_{i,r} - eta * lambda * (w_{i,r} - theta_i)
            # Note: The gradient of the envelope function approx is lambda * (w - theta)
            grad_envelope = self.config.lambda_val * (self.w_local - self.theta)
            self.w_local = self.w_local - self.config.lr * grad_envelope
            
        return self.w_local, self.theta

class Server:
    def __init__(self, config):
        self.config = config
        self.w_global = np.zeros(config.dimension)
        self.clients = []
        
    def add_client(self, client):
        self.clients.append(client)
        
    def aggregate(self, client_models):
        # pFedMe Global Update:
        # w_{t+1} = (1 - beta) * w_t + beta * (sum(w_{i,R}) / S)
        if not client_models:
            return
            
        avg_client_model = np.mean(client_models, axis=0)
        self.w_global = (1 - self.config.beta) * self.w_global + self.config.beta * avg_client_model

    def train(self):
        loss_history = []
        print(f"Starting Training for {self.config.num_global_rounds} rounds...")
        
        for t in range(self.config.num_global_rounds):
            local_weights = []
            personal_losses = []
            
            # Simulate client sampling (using all for demo)
            active_clients = self.clients
            
            for client in active_clients:
                w_local_final, theta_final = client.local_training(self.w_global)
                local_weights.append(w_local_final)
                
                # Monitor personalization loss
                loss = client.loss_function(theta_final, np.arange(len(client.y)))
                personal_losses.append(loss)
            
            # Server update
            self.aggregate(local_weights)
            
            avg_loss = np.mean(personal_losses)
            loss_history.append(avg_loss)
            
            if (t+1) % 10 == 0:
                print(f"Round {t+1}: Avg Personalized Loss = {avg_loss:.4f}")
                
        return loss_history

def generate_data(num_clients, dimension, data_per_client, condition_number=1.0, noise_level=0.1):
    """
    Generates synthetic regression data.
    condition_number: Controls convexity.
    """
    clients_data = []
    # Ground truth model
    true_w = np.random.randn(dimension)
    
    # Scale vector for ill-conditioning
    # Singular values will range from 1 to sqrt(condition_number)
    # Actually, to make condition number K, we want max_eig/min_eig = K.
    # So range from 1 to sqrt(K) for X gives K for X^T X.
    scale_vector = np.linspace(1, np.sqrt(condition_number), dimension)
    
    for i in range(num_clients):
        # Generate X
        X = np.random.randn(data_per_client, dimension)
        # Apply scaling to columns to affect condition number
        X = X * scale_vector
        
        # y = Xw + epsilon
        y = X @ true_w + np.random.randn(data_per_client) * noise_level
        clients_data.append((X, y))
        
    return clients_data

def run_example_high_convexity():
    print("\n=== Running Example 1: High Convexity (Well-conditioned) ===")
    cfg = Config() # Fixed: removed unexpected argument
    cfg.lr = 0.05
    
    data = generate_data(
        num_clients=cfg.num_clients, 
        dimension=cfg.dimension, 
        data_per_client=cfg.data_per_client, 
        condition_number=1.0, 
        noise_level=0.1
    )
    
    server = Server(cfg)
    for i, (X, y) in enumerate(data):
        server.add_client(Client(i, cfg, X, y, task_type="high_convexity"))
        
    return server.train()

def run_example_low_convexity():
    print("\n=== Running Example 2: Low Convexity (Ill-conditioned) ===")
    cfg = Config()
    cfg.lr = 0.01 # Lower LR for stability on ill-conditioned problem
    
    data = generate_data(
        num_clients=cfg.num_clients, 
        dimension=cfg.dimension, 
        data_per_client=cfg.data_per_client, 
        condition_number=100.0, # High condition number
        noise_level=0.1
    )
    
    server = Server(cfg)
    for i, (X, y) in enumerate(data):
        server.add_client(Client(i, cfg, X, y, task_type="low_convexity"))
        
    return server.train()

if __name__ == "__main__":
    hist_high = run_example_high_convexity()
    hist_low = run_example_low_convexity()
    
    if "DISPLAY" not in os.environ:
        plt.switch_backend('Agg')
        
    plt.figure(figsize=(10, 6))
    plt.plot(hist_high, label="High Convexity (Cond=1)", marker='o', markevery=5)
    plt.plot(hist_low, label="Low Convexity (Cond=100)", marker='x', markevery=5)
    plt.title("pFedMe Convergence: High vs Low Convexity")
    plt.xlabel("Global Rounds")
    plt.ylabel("Avg Personalized Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    
    output_plot = "pfedme_results.png"
    plt.savefig(output_plot)
    print(f"\nSimulation complete. Plot saved to {output_plot}")
