import numpy as np
import copy
import matplotlib.pyplot as plt
import os

def softmax(z):
    # Stabilized softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

class Config:
    def __init__(self, num_clients=20, num_global_rounds=200, num_local_rounds=5, 
                 batch_size=20, lambda_val=15.0, lr=0.01, beta=1.0, 
                 k_inner_steps=5, dimension=20, num_classes=5, hidden_size=20):
        self.num_clients = num_clients
        self.num_global_rounds = num_global_rounds
        self.num_local_rounds = num_local_rounds
        self.batch_size = batch_size
        self.lambda_val = lambda_val
        self.lr = lr
        self.beta = beta
        self.k_inner_steps = k_inner_steps
        self.dimension = dimension
        self.num_classes = num_classes
        self.hidden_size = hidden_size

class Client:
    def __init__(self, id, config, X_train, y_train, X_test, y_test, model_type="MLR"):
        self.id = id
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
        
        # Initialize model parameters based on type
        if self.model_type == "MLR":
            # W: (dimension, num_classes)
            # Flattened for optimizer: (dimension * num_classes,)
            self.param_shape = (config.dimension, config.num_classes)
            self.num_params = config.dimension * config.num_classes
        
        elif self.model_type == "DNN":
            # Layer 1: W1 (dimension, hidden)
            # Layer 2: W2 (hidden, num_classes)
            # Biases omitted for simplicity as per common FL benchmarks or treated as part of input
            # We will flatten all params into a single vector
            self.w1_shape = (config.dimension, config.hidden_size)
            self.w2_shape = (config.hidden_size, config.num_classes)
            self.num_params = (config.dimension * config.hidden_size) + (config.hidden_size * config.num_classes)
            
        self.w_local = np.random.randn(self.num_params) * 0.01
        self.theta = np.copy(self.w_local)
        
    def get_gradients(self, theta_flat, X, y):
        # Helper to compute gradients for a batch
        m = len(y)
        y_oh = one_hot(y, self.config.num_classes)
        
        if self.model_type == "MLR":
            # Unpack
            W = theta_flat.reshape(self.param_shape)
            
            # Forward
            logits = X @ W
            probs = softmax(logits)
            
            # Gradient: X.T @ (probs - y_oh) / m + L2 reg ?
            # Note: The proximal term handles heavy lifting, but stronger convexity often includes L2 in f_i.
            # Plan said: CrossEntropy + L2.
            # Grad = (1/m) * X.T(P - Y) + lambda_l2 * W
            diff = probs - y_oh
            grad_W = (X.T @ diff) / m 
            
            # We can add a small native L2 regularization if desired, but pFedMe adds ||theta - w||^2.
            # "Strongly convex" usually implies f_i itself is strongly convex.
            # Adding L2 to f_i:
            l2_reg_strength = 0.001 
            grad_W += l2_reg_strength * W
            
            return grad_W.flatten()
            
        elif self.model_type == "DNN":
            # Unpack
            size_w1 = self.w1_shape[0] * self.w1_shape[1]
            W1 = theta_flat[:size_w1].reshape(self.w1_shape)
            W2 = theta_flat[size_w1:].reshape(self.w2_shape)
            
            # Forward
            Z1 = X @ W1
            A1 = relu(Z1)
            Z2 = A1 @ W2
            probs = softmax(Z2)
            
            # Backprop
            delta2 = probs - y_oh
            dW2 = (A1.T @ delta2) / m
            
            delta1 = (delta2 @ W2.T) * relu_deriv(Z1)
            dW1 = (X.T @ delta1) / m
            
            return np.concatenate((dW1.flatten(), dW2.flatten()))
            
        return np.zeros_like(theta_flat)

    def loss_function(self, theta_flat, X, y):
        # Just Cross Entropy (+ L2 if MLR)
        probs = self.predict_probs(theta_flat, X)
        m = len(y)
        correct_logprobs = -np.log(probs[np.arange(m), y] + 1e-9)
        loss = np.sum(correct_logprobs) / m
        
        if self.model_type == "MLR":
             l2_reg_strength = 0.001
             W = theta_flat.reshape(self.param_shape)
             loss += 0.5 * l2_reg_strength * np.sum(W**2)
             
        return loss

    def predict_probs(self, theta_flat, X):
        if self.model_type == "MLR":
            W = theta_flat.reshape(self.param_shape)
            return softmax(X @ W)
        elif self.model_type == "DNN":
            size_w1 = self.w1_shape[0] * self.w1_shape[1]
            W1 = theta_flat[:size_w1].reshape(self.w1_shape)
            W2 = theta_flat[size_w1:].reshape(self.w2_shape)
            Z1 = X @ W1
            A1 = relu(Z1)
            Z2 = A1 @ W2
            return softmax(Z2)
        return np.zeros((len(X), self.config.num_classes))

    def evaluate(self, theta_flat, X, y):
        probs = self.predict_probs(theta_flat, X)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y)
        loss = self.loss_function(theta_flat, X, y)
        return loss, acc

    def find_personalized_theta(self, w_ref):
        theta_curr = np.copy(self.w_local)
        
        # Use simple full batch for inner steps or mini-batch
        # Using full local train data for simplicity
        indices = np.arange(len(self.y_train))
        
        for k in range(self.config.k_inner_steps):
            grad_f = self.get_gradients(theta_curr, self.X_train, self.y_train)
            grad_prox = self.config.lambda_val * (theta_curr - w_ref)
            theta_curr = theta_curr - self.config.lr * (grad_f + grad_prox)
            
        return theta_curr

    def local_training(self, w_global):
        self.w_local = np.copy(w_global)
        
        for r in range(self.config.num_local_rounds):
            self.theta = self.find_personalized_theta(self.w_local)
            
            # pFedMe update
            grad_envelope = self.config.lambda_val * (self.w_local - self.theta)
            self.w_local = self.w_local - self.config.lr * grad_envelope
            
        return self.w_local, self.theta

class Server:
    def __init__(self, config):
        self.config = config
        # Global model init depends on model type, decided by first client usually or passed config
        # We'll determine num_params from config
        if config.hidden_size > 0: # Proxy for DNN
             num_params = (config.dimension * config.hidden_size) + (config.hidden_size * config.num_classes)
        else: # MLR
             num_params = config.dimension * config.num_classes
             
        self.w_global = np.random.randn(num_params) * 0.01
        self.clients = []
        
    def add_client(self, client):
        self.clients.append(client)
        # Resize global if needed (for safety, though we assume consistent config)
        if len(self.w_global) != client.num_params:
            self.w_global = np.zeros(client.num_params)

    def aggregate(self, client_models):
        if not client_models:
            return
        avg_client_model = np.mean(client_models, axis=0)
        self.w_global = (1 - self.config.beta) * self.w_global + self.config.beta * avg_client_model

    def train(self):
        train_loss_hist = []
        test_acc_hist = []
        
        print(f"Starting Training for {self.config.num_global_rounds} rounds...")
        
        for t in range(self.config.num_global_rounds):
            local_weights = []
            
            # Metrics for this round
            round_train_losses = []
            round_test_accs = []
            
            for client in self.clients:
                # Local Update
                w_local_final, theta_final = client.local_training(self.w_global)
                local_weights.append(w_local_final)
                
                # Evaluation (using Personalized Model theta)
                # "Personalized FL" -> Evaluate personalized models
                tr_loss, _ = client.evaluate(theta_final, client.X_train, client.y_train)
                _, te_acc = client.evaluate(theta_final, client.X_test, client.y_test)
                
                round_train_losses.append(tr_loss)
                round_test_accs.append(te_acc)
            
            # Server Aggregation
            self.aggregate(local_weights)
            
            avg_tr_loss = np.mean(round_train_losses)
            avg_te_acc = np.mean(round_test_accs)
            
            train_loss_hist.append(avg_tr_loss)
            test_acc_hist.append(avg_te_acc)
            
            if (t+1) % 10 == 0:
                print(f"Round {t+1}: Train Loss={avg_tr_loss:.4f}, Test Acc={avg_te_acc:.4f}")
                
        return train_loss_hist, test_acc_hist

def generate_classification_data(num_clients, dimension, num_classes, data_per_client, condition_number=1.0):
    """
    Generates synthetic classification data.
    Uses a simple linear generative model y = argmax(Wx) then adds noise/flip labels or uses clusters.
    To control convexity, we really control the feature spread/conditioning.
    """
    clients_data = []
    
    # Random true weight
    W_true = np.random.randn(dimension, num_classes)
    
    # Scale vector for ill-conditioning
    scale = np.linspace(1, np.sqrt(condition_number), dimension)
    
    for i in range(num_clients):
        # Generate X
        X = np.random.randn(data_per_client, dimension)
        X = X * scale # Ill-conditioning
        
        # Generate Y
        logits = X @ W_true
        # Add noise
        logits += np.random.randn(data_per_client, num_classes) * 0.5
        y = np.argmax(logits, axis=1)
        
        # Split 75/25
        n_train = int(0.75 * data_per_client)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        clients_data.append((X_train, y_train, X_test, y_test))
        
    return clients_data

def run_mlr_experiment():
    print("\n=== Running Strongly Convex (MLR) Experiment ===")
    cfg = Config(num_global_rounds=800, dimension=20, num_classes=5, hidden_size=0) # Hidden=0 -> MLR
    cfg.lr = 0.05
    
    data = generate_classification_data(
        cfg.num_clients, cfg.dimension, cfg.num_classes, 
        data_per_client=100, condition_number=1.0
    )
    
    server = Server(cfg)
    for i, (Xt, yt, Xv, yv) in enumerate(data):
        server.add_client(Client(i, cfg, Xt, yt, Xv, yv, model_type="MLR"))
        
    return server.train()

def run_dnn_experiment():
    print("\n=== Running Non-Convex (DNN) Experiment ===")
    # DNN: Hidden=20, ReLU, Softmax
    cfg = Config(num_global_rounds=800, dimension=20, num_classes=5, hidden_size=20)
    cfg.lr = 0.05
    
    # Use ill-conditioned data to make it harder? Or just standard?
    # User asked for "Non-Convex" case. Neural net is inherently non-convex.
    data = generate_classification_data(
        cfg.num_clients, cfg.dimension, cfg.num_classes, 
        data_per_client=100, condition_number=10.0
    )
    
    server = Server(cfg)
    for i, (Xt, yt, Xv, yv) in enumerate(data):
        server.add_client(Client(i, cfg, Xt, yt, Xv, yv, model_type="DNN"))
        
    return server.train()

if __name__ == "__main__":
    # 1. Strongly Convex (MLR)
    loss_mlr, acc_mlr = run_mlr_experiment()
    
    # 2. Non-Convex (DNN)
    loss_dnn, acc_dnn = run_dnn_experiment()
    
    if "DISPLAY" not in os.environ:
        plt.switch_backend('Agg')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Training Loss
    ax1.plot(loss_mlr, label="MLR (Convex)")
    ax1.plot(loss_dnn, label="DNN (Non-Convex)")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True)
    
    # Plot Test Accuracy
    ax2.plot(acc_mlr, label="MLR (Convex)")
    ax2.plot(acc_dnn, label="DNN (Non-Convex)")
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    output_plot = "pfedme_classification_results.png"
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"\nSimulation complete. Results saved to {output_plot}")
