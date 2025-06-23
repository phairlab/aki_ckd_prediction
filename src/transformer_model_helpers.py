import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

# # Custom Dataset for tabular data
# class KidneyDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X.values, dtype=torch.float32)
#         self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# # Adjusted Transformer model for tabular data with 31 input features and smaller hidden sizes
# class TabularTransformer(nn.Module):
#     def __init__(self, num_features=31, d_model=16, nhead=2, num_layers=1, dim_feedforward=32):
#         super().__init__()
#         self.input_proj = nn.Linear(num_features, d_model)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.regressor = nn.Linear(d_model, 1)

#     def forward(self, x):
#         # x: (batch_size, num_features)
#         x = self.input_proj(x).unsqueeze(1)  # (batch_size, seq_len=1, d_model)
#         x = self.transformer(x)              # (batch_size, seq_len=1, d_model)
#         x = x.squeeze(1)                     # (batch_size, d_model)
#         return self.regressor(x)             # (batch_size, 1)
    
# def get_dataloader(X, y, batch_size=32, shuffle=True):
#     dataset = KidneyDataset(X, y)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# # Training loop
# def train_model(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     for X_batch, y_batch in dataloader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * X_batch.size(0)
#     return total_loss / len(dataloader.dataset)

# # Validation loop
# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for X_batch, y_batch in dataloader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             total_loss += loss.item() * X_batch.size(0)
#     return total_loss / len(dataloader.dataset)


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


class LargeTabularTransformer(nn.Module):
    """
    LargeTabularTransformer: A transformer model optimized for high-dimensional tabular data.
    
    This model is designed for datasets with many features (~1500) and moderate sample sizes (~4000).
    It incorporates dimensionality reduction, regularization techniques, and an efficient 
    transformer architecture to handle high-dimensional input effectively.
    
    Attributes:
        embedding_dim (int): Dimension of the embedding space (default: 128)
        bottleneck_dim (int): Dimension of bottleneck layer for dimensionality reduction (default: 256)
        num_heads (int): Number of attention heads in the transformer (default: 8)
        num_layers (int): Number of transformer encoder layers (default: 3)
        dropout (float): Dropout rate used throughout the model (default: 0.2)
        
    Args:
        input_dim (int): Number of input features
        num_classes (int, optional): Number of output classes. Defaults to 2 (binary classification).
        
    Example:
        >>> model = LargeTabularTransformer(input_dim=1500, num_classes=2)
        >>> inputs = torch.randn(64, 1500)  # batch_size=64, features=1500
        >>> logits = model(inputs)  # Shape: [64, 2]
    """
    
    def __init__(self, input_dim, num_classes=2):
        super(LargeTabularTransformer, self).__init__()
        self.embedding_dim = 128
        self.bottleneck_dim = 256
        self.num_heads = 8
        self.num_layers = 3
        self.dropout = 0.2
        
        # Dimensionality reduction via bottleneck
        self.feature_reduction = nn.Sequential(
            nn.BatchNorm1d(input_dim),  # Normalize inputs
            nn.Linear(input_dim, self.bottleneck_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout),
            nn.Linear(self.bottleneck_dim, self.embedding_dim)
        )
        
        # Layer normalization before transformer for stable training
        self.pre_transformer_norm = nn.LayerNorm(self.embedding_dim)
        
        # Transformer encoder with increased capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embedding_dim*4,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu'  # GELU activation often works better for transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.embedding_dim)
        )
        
        # More sophisticated classification head with residual connections
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4),
            nn.GELU(),
            nn.Dropout(self.dropout/2),  # Less dropout in final layers
            nn.Linear(self.embedding_dim // 4, num_classes)
        )
        
    def forward(self, x):
        # Initial dimensionality reduction
        x = self.feature_reduction(x)  # [batch_size, embedding_dim]
        x = self.pre_transformer_norm(x)
        
        # Add sequence dimension and apply transformer
        x = x.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        x = self.transformer_encoder(x)  # [batch_size, 1, embedding_dim]
        x = x.squeeze(1)  # [batch_size, embedding_dim]
        
        # Classification
        logits = self.classifier(x)  # [batch_size, num_classes]
        return logits


# def train_transformer_model(X_train, y_train, device, epochs=20, batch_size=32):
#     """
#     Train a TabularTransformer model on the given training data.
#     This function initializes a TabularTransformer model and trains it using the provided
#     training data. It converts numpy arrays to PyTorch tensors, creates a DataLoader for
#     batch processing, and uses the AdamW optimizer with CrossEntropyLoss.
#     Parameters
#     ----------
#     X_train : numpy.ndarray
#         Features of the training dataset with shape (n_samples, n_features)
#     y_train : numpy.ndarray
#         Target labels for the training dataset with shape (n_samples,)
#     epochs : int, optional
#         Number of training epochs, by default 20
#     batch_size : int, optional
#         Size of batches for training, by default 32
#     Returns
#     -------
#     TabularTransformer
#         The trained PyTorch model
#     Notes
#     -----
#     The current implementation does not use the validation data (X_val, y_val)
#     even if provided. Progress is printed every 5 epochs.
#     """
#     # Scale the input data to help with numerical stability
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)

#     # Convert data to PyTorch tensors
#     X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    
#     # Create dataset and dataloader
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
#     # Initialize model and move to GPU
#     model = TabularTransformer(input_dim=X_train.shape[1]).to(device)
    
#     # Apply weight initialization to help with training stability
#     def init_weights(m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#     model.apply(init_weights)
    
#     # Loss function and optimizer with a lower learning rate
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    
#     # Learning rate scheduler to reduce LR over time
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=2, verbose=True
#     )
    
#     # Training loop
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
            
#             # Check for NaN values in outputs
#             if torch.isnan(outputs).any():
#                 print("Warning: NaN detected in model outputs")
#                 continue
                
#             loss = criterion(outputs, labels)
            
#             # Skip backward pass if loss is NaN
#             if torch.isnan(loss):
#                 print(f"Warning: NaN loss encountered in epoch {epoch+1}")
#                 continue
                
#             loss.backward()
            
#             # Gradient clipping to prevent exploding gradients
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
#             running_loss += loss.item()
        
#         epoch_loss = running_loss / len(train_loader)
#         # Update learning rate based on loss
#         scheduler.step(epoch_loss)
        
#         # Print progress
#         if (epoch+1) % 5 == 0:
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
#     # Store the scaler in the model for later use in prediction
#     model.scaler = scaler
#     return model


# Define a function to train with validation and early stopping
def train_with_validation(X_train, y_train, device, epochs=20, batch_size=32, 
                         validation_split=0.15, early_stopping=5, learning_rate=1e-4):
    """
    Train a transformer model with validation-based early stopping.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Features of the training dataset
    y_train : numpy.ndarray
        Target labels for the training dataset
    device : torch.device
        Device to use for training (CPU or GPU)
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training
    validation_split : float
        Fraction of training data to use for validation
    early_stopping : int
        Number of epochs with no improvement after which training will be stopped
    learning_rate : float
        Learning rate for the optimizer
        
    Returns:
    --------
    model : TabularTransformer
        The trained model
    train_losses : list
        Training losses per epoch
    val_losses : list
        Validation losses per epoch
    best_epoch : int
        Epoch with the best validation loss
    """
    
    # Split training data into train and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=1202, stratify=y_train
    )
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_split, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TabularTransformer(input_dim=X_train.shape[1]).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    train_losses = []
    val_losses = []
    best_epoch = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bes_model_state = model.state_dict().copy()
            best_epoch = epoch
            counter = 0
        else:
            counter += 1
            
        # Early stopping check
        if counter >= early_stopping:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, best_epoch



def predict_transformer(model, X_test, device):
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
    # Scale the test data using the same scaler used during training
    if hasattr(model, 'scaler'):
        X_test_scaled = model.scaler.transform(X_test)
    else:
        # If no scaler is stored, use the data as is (but print a warning)
        X_test_scaled = X_test
        # print("Warning: No scaler found in model. Using unscaled data for prediction.")

    # Convert data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)
    
    return predictions.cpu().numpy(), probs[:, 1].cpu().numpy()