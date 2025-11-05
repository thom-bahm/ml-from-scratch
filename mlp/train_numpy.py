import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from mlp_numpy import MLP_Numpy, one_hot_encode

def prepare_data_for_numpy(train_loader, test_loader):
    """
    Convert PyTorch DataLoader data to NumPy arrays for the NumPy MLP.
    
    Args:
        train_loader: PyTorch DataLoader for training data
        test_loader: PyTorch DataLoader for test data
    
    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    
    TODO: Implement data conversion
    Steps:
    1. Iterate through DataLoader to collect all batches
    2. Convert PyTorch tensors to NumPy arrays
    3. Flatten images from (batch, 1, 28, 28) to (batch, 784)
    4. Transpose to shape (784, num_samples) for matrix multiplication
    5. Convert labels to NumPy arrays
    
    Hint: Use .numpy() to convert PyTorch tensors
    Hint: Use np.concatenate() to combine batches
    """
    pass


def train_numpy_mlp(model, X_train, y_train, X_val, y_val, 
                    epochs=20, batch_size=64):
    """
    Train the NumPy MLP model.
    
    Args:
        model: MLP_Numpy instance
        X_train: Training data of shape (784, num_train_samples)
        y_train: Training labels of shape (num_train_samples,)
        X_val: Validation data of shape (784, num_val_samples)
        y_val: Validation labels of shape (num_val_samples,)
        epochs: Number of training epochs
        batch_size: Size of mini-batches
    
    Returns:
        train_losses: List of training losses per epoch
        val_accuracies: List of validation accuracies per epoch
    
    TODO: Implement training loop
    Steps for each epoch:
    1. Shuffle training data
    2. Split into mini-batches
    3. For each batch:
       - Convert labels to one-hot
       - Perform training step
       - Track loss
    4. Compute validation accuracy
    5. Print progress
    
    Think about:
    - How to create mini-batches from the data?
    - How to shuffle the data each epoch?
    - When to compute validation accuracy?
    """
    train_losses = []
    val_accuracies = []
    
    num_samples = X_train.shape[1]
    num_batches = num_samples // batch_size
    
    for epoch in trange(epochs, desc="Training"):
        epoch_losses = []
        
        # TODO: Shuffle training data
        # Hint: Use np.random.permutation() to get shuffled indices
        
        # TODO: Mini-batch training
        # For each batch:
        #   1. Get batch data (X_batch, y_batch)
        #   2. Convert y_batch to one-hot (Y_batch)
        #   3. Perform training step
        #   4. Record loss
        
        # Compute average loss for the epoch
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        train_losses.append(avg_loss)
        
        # TODO: Compute validation accuracy
        # Y_val = one_hot_encode(y_val, 10)
        # val_acc = model.compute_accuracy(X_val, Y_val)
        # val_accuracies.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%")
    
    return train_losses, val_accuracies


def plot_training_results(train_losses, val_accuracies):
    """
    Plot training loss and validation accuracy curves.
    
    TODO: Implement plotting
    Create two subplots:
    1. Training loss over epochs
    2. Validation accuracy over epochs
    """
    pass


if __name__ == "__main__":
    print("Training NumPy MLP on MNIST...")
    
    # TODO: Implement the main training script
    # Steps:
    # 1. Load MNIST data using your existing data_utils
    # 2. Convert data to NumPy format
    # 3. Create MLP model with appropriate architecture
    # 4. Train the model
    # 5. Plot results
    # 6. Save the model weights (using np.save)
    
    # Example structure:
    """
    import data_utils as du
    
    # Load data
    train_loader, test_loader = du.load_mnist_data(batch_size=64)
    
    # Convert to NumPy
    X_train, y_train, X_test, y_test = prepare_data_for_numpy(train_loader, test_loader)
    
    # Create model
    layer_sizes = [784, 512, 256, 128, 10]
    model = MLP_Numpy(layer_sizes, learning_rate=0.01)
    
    # Train
    train_losses, val_accs = train_numpy_mlp(
        model, X_train, y_train, X_test, y_test,
        epochs=20, batch_size=64
    )
    
    # Plot results
    plot_training_results(train_losses, val_accs)
    
    # Test final accuracy
    Y_test = one_hot_encode(y_test, 10)
    final_acc = model.compute_accuracy(X_test, Y_test)
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    
    # Save model
    np.save('mlp_numpy_weights.npy', {
        'weights': model.weights,
        'biases': model.biases,
        'layer_sizes': model.layer_sizes
    })
    """
    
    print("\nImplement the training script above to get started!")
