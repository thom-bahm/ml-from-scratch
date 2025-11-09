import numpy as np
from typing import List, Tuple

class MLP_Numpy:
    """
    Multi-Layer Perceptron implemented from scratch using only NumPy.
    
    This implementation will help you understand:
    - How weights and biases are stored and initialized
    - Forward propagation through multiple layers
    - Backward propagation and gradient computation
    - Weight updates using gradient descent
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        """
        Initialize the MLP with given layer sizes.
        
        Args:
            layer_sizes: List of integers representing the size of each layer
                        Example: [784, 512, 256, 128, 10] for MNIST
                        - 784: input size (28x28 flattened)
                        - 512, 256, 128: hidden layer sizes
                        - 10: output size (number of classes)
            learning_rate: Learning rate for gradient descent
        
        TODO: Initialize the following:
        - self.layer_sizes: Store the layer sizes
        - self.learning_rate: Store the learning rate
        - self.weights: List of weight matrices for each layer
        - self.biases: List of bias vectors for each layer
        - self.num_layers: Number of layers (excluding input)
        
        Think about:
        - What shape should each weight matrix be?
        - How should you initialize the weights? (random, zeros, Xavier, He?)
        - What shape should each bias vector be?
        - How should you initialize the biases?
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # TODO: Initialize weights and biases for each layer
        # Hint: For layer i, weight matrix shape should be (layer_sizes[i+1], layer_sizes[i])
        # Hint: For layer i, bias vector shape should be (layer_sizes[i+1], 1)
        # Hint: Consider using Xavier initialization: np.random.randn() * np.sqrt(2.0 / layer_sizes[i])
        for i in range(self.num_layers):
            # He initialization for weights
            self.weights.append(
                np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            )
            # Zeros for biases
            self.biases.append(
                np.zeros((layer_sizes[i+1], 1))
            )
        
    
    def relu(self, Z: np.ndarray) -> np.ndarray:
        """
        ReLU activation function: f(x) = max(0, x)
        
        Args:
            Z: Input array of any shape
        
        Returns:
            Output array with ReLU applied element-wise
        
        TODO: Implement ReLU activation
        Hint: Use np.maximum()
        """
        return np.maximum(np.zeros(Z.shape), Z)
    
    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU: f'(x) = 1 if x > 0, else 0
        
        Args:
            Z: Input array (usually the pre-activation values)
        
        Returns:
            Derivative array
        
        TODO: Implement ReLU derivative
        Hint: Return 1 where Z > 0, else 0
        """
        return (Z > 0).astype(float)
    
    def softmax(self, Z: np.ndarray) -> np.ndarray:
        """
        Softmax activation function for output layer.
        Converts raw scores to probabilities that sum to 1.
        
        Args:
            Z: Input array of shape (num_classes, batch_size)
        
        Returns:
            Probabilities of shape (num_classes, batch_size)
        
        TODO: Implement softmax
        Formula: softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
        
        Hint: For numerical stability, subtract max(Z) before computing exp
        Hint: Use np.exp(), np.sum() with appropriate axis
        """
        probs = []
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        probs = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return probs
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (input_size, batch_size)
               For MNIST: (784, batch_size)
        
        Returns:
            A: Output activations of shape (output_size, batch_size)
            cache: Dictionary containing intermediate values needed for backprop
                   Should store: 'Z' values (pre-activation) and 'A' values (post-activation)
                   for each layer
        
        TODO: Implement forward pass
        Steps for each layer:
        1. Compute Z = W @ A_prev + b  (linear transformation)
        2. Compute A = activation(Z)   (apply activation function)
        3. Store Z and A in cache for backprop
        
        Think about:
        - What activation function to use for hidden layers? (ReLU)
        - What activation function to use for output layer? (Softmax for classification)
        - How to store intermediate values for backpropagation?
        """
        cache = {'A0': X}  # Store input as A0
        A = X
        
        # TODO: Forward pass through all layers
        # For layers 1 to num_layers-1, use ReLU
        # For the last layer, use softmax
        for l in range(self.num_layers):
            Z = self.weights[l] @ A + self.biases[l]
            cache[f'Z{l+1}'] = Z
            
            if l == self.num_layers - 1:
                # Output layer with softmax
                A = self.softmax(Z)
            else:
                # Hidden layers with ReLU
                A = self.relu(Z)
            
            cache[f'A{l+1}'] = A
        return A, cache
    
    def backward(self, Y: np.ndarray, cache: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation to compute gradients.
        
        Args:
            Y: True labels of shape (output_size, batch_size)
               For MNIST: (10, batch_size) in one-hot encoding
            cache: Dictionary from forward pass containing Z's and A's
        
        Returns:
            dW: List of weight gradients for each layer
            db: List of bias gradients for each layer
        
        TODO: Implement backpropagation using chain rule
        
        Key formulas:
        - Output layer: dZ_L = A_L - Y  (when using softmax + cross-entropy)
        - Hidden layers: dZ_l = (W_{l+1}^T @ dZ_{l+1}) * activation_derivative(Z_l)
        - Weight gradient: dW_l = (1/m) * dZ_l @ A_{l-1}^T
        - Bias gradient: db_l = (1/m) * sum(dZ_l, axis=1, keepdims=True)
        
        Where m is the batch size
        
        Think about:
        - How to work backwards through the layers?
        - What's the gradient of softmax + cross-entropy loss? (Hint: it's simple!)
        - How to apply the chain rule for hidden layers?
        """
        m = Y.shape[1]  # batch size
        dW = []
        db = []
        
        # TODO: Implement backpropagation
        # Start from the output layer and work backwards
        
        pass
    
    def update_parameters(self, dW: List[np.ndarray], db: List[np.ndarray]):
        """
        Update weights and biases using gradient descent.
        
        Args:
            dW: List of weight gradients
            db: List of bias gradients
        
        TODO: Implement parameter update
        Formula: W = W - learning_rate * dW
                 b = b - learning_rate * db
        """
        pass
    
    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Perform one training step: forward pass, compute loss, backward pass, update.
        
        Args:
            X: Input data of shape (input_size, batch_size)
            Y: True labels of shape (output_size, batch_size) - one-hot encoded
        
        Returns:
            loss: Cross-entropy loss for this batch
        
        TODO: Implement training step
        Steps:
        1. Forward pass
        2. Compute loss
        3. Backward pass
        4. Update parameters
        5. Return loss
        """
        # Forward pass
        A, cache = self.forward(X)
        
        # Compute loss (cross-entropy)
        # TODO: Implement cross-entropy loss
        # Formula: L = -(1/m) * sum(Y * log(A))
        # Hint: Add small epsilon (1e-8) to log to avoid log(0)
        
        # Backward pass
        dW, db = self.backward(Y, cache)
        
        # Update parameters
        self.update_parameters(dW, db)
        
        # Return loss
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input data of shape (input_size, batch_size)
        
        Returns:
            predictions: Predicted class indices of shape (batch_size,)
        
        TODO: Implement prediction
        Steps:
        1. Forward pass
        2. Find the class with highest probability for each sample
        Hint: Use np.argmax()
        """
        pass
    
    def compute_accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute accuracy on given data.
        
        Args:
            X: Input data of shape (input_size, batch_size)
            Y: True labels of shape (output_size, batch_size) - one-hot encoded
        
        Returns:
            accuracy: Percentage of correct predictions
        
        TODO: Implement accuracy computation
        Steps:
        1. Get predictions
        2. Convert one-hot Y to class indices
        3. Compare and compute accuracy
        """
        pass


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        y: Integer labels of shape (batch_size,)
        num_classes: Total number of classes
    
    Returns:
        One-hot encoded array of shape (num_classes, batch_size)
    
    TODO: Implement one-hot encoding
    Example: y = [1, 0, 3] with num_classes=4
    Returns: [[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0],
              [0, 0, 1]]
    
    Hint: Create a zero matrix and set appropriate positions to 1
    """
    pass


if __name__ == "__main__":
    """
    Test your implementation with a simple example
    """
    print("Testing NumPy MLP implementation...")
    
    # Create a small MLP
    layer_sizes = [784, 128, 64, 10]  # For MNIST
    model = MLP_Numpy(layer_sizes, learning_rate=0.01)
    # Create some dummy data to test
    batch_size = 32
    X_test = np.random.randn(784, batch_size)
    y_test = np.random.randint(0, 10, batch_size)
    Y_test = one_hot_encode(y_test, 10)
    
    print(X_test)
    print(model.relu(X_test))
    # Test forward pass
    # output, cache = model.forward(X_test)
    # print(f"Output shape: {output.shape}")  # Should be (10, 32)
    
    # Test training step
    # loss = model.train_step(X_test, Y_test)
    # print(f"Loss: {loss}")
    
    # Test prediction
    # predictions = model.predict(X_test)
    # print(f"Predictions shape: {predictions.shape}")  # Should be (32,)
    
    # Test accuracy
    # accuracy = model.compute_accuracy(X_test, Y_test)
    # print(f"Accuracy: {accuracy}%")
    
    print("\nUncomment the test code above once you implement the methods!")
