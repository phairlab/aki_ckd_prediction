from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


class TabularTransformer(nn.Module):
    """
    TabularTransformer: A PyTorch model that applies transformer architecture to tabular data.
    This model converts tabular input features into an embedding space and processes them using
    a transformer encoder architecture, followed by a classification head to make predictions.
    Attributes:
        embedding_dim (int): Dimension of the embedding space (default: 64)
        num_heads (int): Number of attention heads in the transformer (default: 4)
        num_layers (int): Number of transformer encoder layers (default: 2)
        dropout (float): Dropout rate used in transformer and classifier (default: 0.1)
        input_embedding (nn.Linear): Linear layer to project input features to embedding space
        transformer_encoder (nn.TransformerEncoder): Stack of transformer encoder layers
        classifier (nn.Sequential): Classification head to produce class logits
    Args:
        input_dim (int): Number of input features
        num_classes (int, optional): Number of output classes. Defaults to 2 (binary classification).
    Example:
        >>> model = TabularTransformer(input_dim=10, num_classes=3)
        >>> inputs = torch.randn(32, 10)  # batch_size=32, features=10
        >>> logits = model(inputs)  # Shape: [32, 3]
    """

    def __init__(self, input_dim, num_classes=2):
        super(TabularTransformer, self).__init__()
        self.embedding_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.1
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, self.embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embedding_dim*4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [batch_size, seq_len]
        x = self.input_embedding(x)  # [batch_size, embedding_dim]
        x = x.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, embedding_dim]
        x = self.transformer_encoder(x)  # [batch_size, 1, embedding_dim]
        x = x.squeeze(1)  # Remove sequence dimension: [batch_size, embedding_dim]
        logits = self.classifier(x)  # [batch_size, num_classes]
        return logits


def train_transformer_model(X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32):
    """
    Train a TabularTransformer model on the given training data.
    This function initializes a TabularTransformer model and trains it using the provided
    training data. It converts numpy arrays to PyTorch tensors, creates a DataLoader for
    batch processing, and uses the AdamW optimizer with CrossEntropyLoss.
    Parameters
    ----------
    X_train : numpy.ndarray
        Features of the training dataset with shape (n_samples, n_features)
    y_train : numpy.ndarray
        Target labels for the training dataset with shape (n_samples,)
    X_val : numpy.ndarray, optional
        Features of the validation dataset, by default None (not used in current implementation)
    y_val : numpy.ndarray, optional
        Target labels for the validation dataset, by default None (not used in current implementation)
    epochs : int, optional
        Number of training epochs, by default 20
    batch_size : int, optional
        Size of batches for training, by default 32
    Returns
    -------
    TabularTransformer
        The trained PyTorch model
    Notes
    -----
    The current implementation does not use the validation data (X_val, y_val)
    even if provided. Progress is printed every 5 epochs.
    """


    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = TabularTransformer(input_dim=X_train.shape[1])
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Print progress
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    return model


def predict_transformer(model, X_test):
    """
    Generate predictions using a trained transformer model.
    Parameters:
    -----------
    model : torch.nn.Module
        The trained transformer model.
    X_test : numpy.ndarray
        The input test data.
    Returns:
    --------
    tuple
        A tuple containing:
        - numpy.ndarray: The predicted class indices.
        - numpy.ndarray: The probabilities for the positive class (class 1).
    Notes:
    ------
    This function converts the input data to PyTorch tensors, 
    puts the model in evaluation mode, and generates class predictions
    along with probabilities using softmax.
    """

    # Convert data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)
    
    return predictions.numpy(), probs[:, 1].numpy()