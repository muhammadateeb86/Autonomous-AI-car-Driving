import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import torch.cuda as cuda

# Set random seed for reproducibility
torch.manual_seed(42)
if cuda.is_available():
    torch.cuda.manual_seed(42)

# Define the MLP model with mixed outputs
class TORCS_MLP(nn.Module):
    def __init__(self, input_size=71, hidden_sizes=[256, 128, 64]):
        super(TORCS_MLP, self).__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Dropout(0.3)
        )
        # Output heads
        self.gear_head = nn.Linear(hidden_sizes[2], 7)  # 7 classes for gear
        self.brake_head = nn.Linear(hidden_sizes[2], 2)  # 2 classes for brake
        self.accel_head = nn.Linear(hidden_sizes[2], 1)  # Regression for accel
        self.steer_head = nn.Linear(hidden_sizes[2], 1)  # Regression for steer
    
    def forward(self, x):
        features = self.backbone(x)
        gear = self.gear_head(features)  # Logits for classification
        brake = self.brake_head(features)  # Logits for classification
        accel = torch.sigmoid(self.accel_head(features))  # [0, 1] range
        steer = torch.tanh(self.steer_head(features))  # [-1, 1] range
        return gear, brake, accel, steer

# Load and preprocess the dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found")
    
    df = pd.read_csv(file_path)
    df = df.drop(columns=['timestamp'])  # Drop timestamp
    
    # Define input and output columns
    output_cols = ['gear', 'brake', 'accel', 'steer']
    input_cols = [col for col in df.columns if col not in output_cols]
    
    # Validate dataset
    if not all(col in df.columns for col in input_cols + output_cols):
        raise ValueError("Dataset missing required columns")
    
    # Split features and targets
    X = df[input_cols].values
    y_gear = df['gear'].values
    y_brake = df['brake'].values
    y_accel = df['accel'].values
    y_steer = df['steer'].values
    
    # Map gear to class indices ({-1, 1, 2, 3, 4, 5, 6} -> {0, 1, 2, 3, 4, 5, 6})
    gear_mapping = {-1: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    y_gear = np.array([gear_mapping[g] for g in y_gear])
    
    # Normalize features
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    # Normalize accel and steer
    scaler_accel = StandardScaler()
    y_accel = scaler_accel.fit_transform(y_accel.reshape(-1, 1)).flatten()
    scaler_steer = StandardScaler()
    y_steer = scaler_steer.fit_transform(y_steer.reshape(-1, 1)).flatten()
    
    return X, (y_gear, y_brake, y_accel, y_steer), scaler_X, (scaler_accel, scaler_steer)

# Create PyTorch datasets
def create_datasets(X, y, test_size=0.2):
    y_gear, y_brake, y_accel, y_steer = y
    X_train, X_val, y_gear_train, y_gear_val, y_brake_train, y_brake_val, y_accel_train, y_accel_val, y_steer_train, y_steer_val = train_test_split(
        X, y_gear, y_brake, y_accel, y_steer, test_size=test_size, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_gear_train = torch.LongTensor(y_gear_train)
    y_brake_train = torch.LongTensor(y_brake_train)
    y_accel_train = torch.FloatTensor(y_accel_train)
    y_steer_train = torch.FloatTensor(y_steer_train)
    
    X_val = torch.FloatTensor(X_val)
    y_gear_val = torch.LongTensor(y_gear_val)
    y_brake_val = torch.LongTensor(y_brake_val)
    y_accel_val = torch.FloatTensor(y_accel_val)
    y_steer_val = torch.FloatTensor(y_steer_val)
    
    train_dataset = TensorDataset(X_train, y_gear_train, y_brake_train, y_accel_train, y_steer_train)
    val_dataset = TensorDataset(X_val, y_gear_val, y_brake_val, y_accel_val, y_steer_val)
    
    return train_dataset, val_dataset

# Training function
def train_model(model, train_loader, val_loader, epochs=100, patience=10, device='cuda'):
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_gear, y_brake, y_accel, y_steer in train_loader:
            X_batch = X_batch.to(device)
            y_gear, y_brake, y_accel, y_steer = y_gear.to(device), y_brake.to(device), y_accel.to(device), y_steer.to(device)
            
            optimizer.zero_grad()
            gear_pred, brake_pred, accel_pred, steer_pred = model(X_batch)
            
            loss_gear = criterion_ce(gear_pred, y_gear)
            loss_brake = criterion_ce(brake_pred, y_brake)
            loss_accel = criterion_mse(accel_pred.squeeze(), y_accel)
            loss_steer = criterion_mse(steer_pred.squeeze(), y_steer)
            
            loss = loss_gear + loss_brake + loss_accel + loss_steer
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_gear, y_brake, y_accel, y_steer in val_loader:
                X_batch = X_batch.to(device)
                y_gear, y_brake, y_accel, y_steer = y_gear.to(device), y_brake.to(device), y_accel.to(device), y_steer.to(device)
                
                gear_pred, brake_pred, accel_pred, steer_pred = model(X_batch)
                
                loss_gear = criterion_ce(gear_pred, y_gear)
                loss_brake = criterion_ce(brake_pred, y_brake)
                loss_accel = criterion_mse(accel_pred.squeeze(), y_accel)
                loss_steer = criterion_mse(steer_pred.squeeze(), y_steer)
                
                loss = loss_gear + loss_brake + loss_accel + loss_steer
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    model.load_state_dict(best_model_state)
    return model, best_val_loss

# Inference function
def predict(model, scaler_X, scaler_accel, scaler_steer, new_data, device='cuda'):
    model.eval()
    gear_mapping = {0: -1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    
    # Ensure new_data is a numpy array with shape (n_samples, 72)
    if isinstance(new_data, pd.DataFrame):
        new_data = new_data.values
    new_data = np.asarray(new_data, dtype=np.float32)
    
    # Normalize inputs
    new_data_scaled = scaler_X.transform(new_data)
    new_data_tensor = torch.FloatTensor(new_data_scaled).to(device)
    
    with torch.no_grad():
        gear_pred, brake_pred, accel_pred, steer_pred = model(new_data_tensor)
        
        # Process outputs
        gear = torch.argmax(gear_pred, dim=1).cpu().numpy()
        gear = np.array([gear_mapping[g] for g in gear])  # Map back to original gear values
        brake = torch.argmax(brake_pred, dim=1).cpu().numpy()  # 0 or 1
        accel = scaler_accel.inverse_transform(accel_pred.cpu().numpy())  # [0, 1]
        steer = scaler_steer.inverse_transform(steer_pred.cpu().numpy())  # [-1, 1]
    
    return gear, brake, accel, steer

def main():
    # Check for GPU
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    file_path = '/sensor_data.csv'  # Update with your CSV path
    try:
        X, y, scaler_X, (scaler_accel, scaler_steer) = load_data(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create datasets and loaders
    train_dataset, val_dataset = create_datasets(X, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = TORCS_MLP()
    
    # Train model
    try:
        model, best_val_loss = train_model(model, train_loader, val_loader, device=device)
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Save model
    model_path = 'torcs_mlp.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save scalers
    scaler_X_path = 'scaler_X.pkl'
    scaler_accel_path = 'scaler_accel.pkl'
    scaler_steer_path = 'scaler_steer.pkl'
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_accel, scaler_accel_path)
    joblib.dump(scaler_steer, scaler_steer_path)
    print(f"Scalers saved to {scaler_X_path}, {scaler_accel_path}, {scaler_steer_path}")
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    # Example inference
    print("\nRunning example inference...")
    sample_data = X[:5]  # Use a few samples from the dataset for testing
    try:
        gear, brake, accel, steer = predict(model, scaler_X, scaler_accel, scaler_steer, sample_data, device)
        print("Sample predictions:")
        for i in range(len(gear)):
            print(f"Sample {i+1}: Gear={gear[i]}, Brake={brake[i]}, Accel={accel[i]:.4f}, Steer={steer[i]:.4f}")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == '__main__':
    main()