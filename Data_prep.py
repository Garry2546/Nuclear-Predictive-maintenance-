import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Device Selection
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------
# 2. Data Loading and Preprocessing Function
# ------------------------------
def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses sensor data for LSTM training.
    Steps:
      - Drop empty columns and those with >60% missing values.
      - Convert and sort timestamps; generate an expected 1-minute interval range.
      - Interpolate and fill missing sensor values.
      - Cap outliers using the Median Absolute Deviation (MAD) method.
      - Generate engineered features: rolling mean, rolling std, and first differences.
      - Encode target labels if available.
      - Scale all selected features.
      - Convert the features and labels into PyTorch tensors.
    
    Returns:
      data_tensor: Tensor of preprocessed features.
      labels_tensor: Tensor of target labels (or None if not available).
      scaler: Fitted MinMaxScaler object.
      feature_cols: List of feature names used.
    """
    # Load data using Pandas
    data = pd.read_csv(file_path)
    
    # Drop completely empty columns
    data = data.dropna(axis=1, how='all')
    
    # Drop columns with more than 60% missing values
    missing_threshold = 0.6
    missing_fraction = data.isna().sum() / len(data)
    cols_to_drop = missing_fraction[missing_fraction > missing_threshold].index
    data = data.drop(columns=cols_to_drop)
    
    # Handle the timestamp column: convert, sort, and resample if needed
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Create an expected timestamp range with 1-minute intervals
        expected_time_range = pd.DataFrame({
            'timestamp': pd.date_range(start=data['timestamp'].min(), 
                                       end=data['timestamp'].max(), 
                                       freq='1min')
        })
        
        # Merge to ensure continuity in timestamps
        data = expected_time_range.merge(data, on='timestamp', how='left')
        
        # Interpolate missing sensor values for smooth transitions, then fallback to forward/backward fill
        data.interpolate(method='linear', inplace=True)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
    
    # Identify sensor columns (excluding non-numeric columns and the target column)
    non_sensor_cols = ['timestamp', 'machine_status']
    sensor_cols = [
        col for col in data.columns 
        if col not in non_sensor_cols and np.issubdtype(data[col].dtype, np.number)
    ]
    
    # ------------------------------
    # Outlier Handling: Cap outliers using Median Absolute Deviation (MAD)
    # ------------------------------
    def cap_outliers(series, factor=3):
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return series
        lower_bound = median - factor * mad
        upper_bound = median + factor * mad
        return series.clip(lower_bound, upper_bound)
    
    for col in sensor_cols:
        data[col] = cap_outliers(data[col])
    
    # ------------------------------
    # Feature Engineering: Rolling statistics and first differences
    # ------------------------------
    for col in sensor_cols:
        data[f'{col}_rolling_mean'] = data[col].rolling(window=10, min_periods=1).mean()
        data[f'{col}_rolling_std'] = data[col].rolling(window=10, min_periods=1).std().fillna(0)
        data[f'{col}_diff'] = data[col].diff().fillna(0)
    
    # Encode target labels (machine_status) if available
    if 'machine_status' in data.columns:
        le = LabelEncoder()
        data['machine_status'] = le.fit_transform(data['machine_status'].astype(str))
    
    # Combine original sensor features with engineered features
    feature_cols = (
        sensor_cols + 
        [f'{col}_rolling_mean' for col in sensor_cols] + 
        [f'{col}_rolling_std' for col in sensor_cols] + 
        [f'{col}_diff' for col in sensor_cols]
    )
    
    # ------------------------------
    # Scaling: Scale the features using MinMaxScaler
    # ------------------------------
    scaler = MinMaxScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    # Convert features and target (if available) to PyTorch tensors
    data_tensor = torch.tensor(data[feature_cols].values, dtype=torch.float32).to(device)
    if 'machine_status' in data.columns:
        labels_tensor = torch.tensor(data['machine_status'].values, dtype=torch.long).to(device)
    else:
        labels_tensor = None
    
    return data_tensor, labels_tensor, scaler, feature_cols

# ------------------------------
# 3. Convert Data into Sequences for LSTM Classification
# ------------------------------
def create_lstm_sequences(data_tensor, labels_tensor, seq_length):
    """
    Convert preprocessed time-series data into overlapping sequences for LSTM classification.
    
    Args:
        data_tensor (torch.Tensor): Tensor of shape (N, num_features) containing the feature data.
        labels_tensor (torch.Tensor): Tensor of shape (N,) containing the target labels.
        seq_length (int): The number of consecutive time steps to include in each sequence.
    
    Returns:
        X_seq (torch.Tensor): Tensor of shape (num_sequences, seq_length, num_features).
        y_seq (torch.Tensor): Tensor of shape (num_sequences,) where each label corresponds to the label at the last time step of the sequence.
    """
    X_seq = []
    y_seq = []
    num_samples = data_tensor.shape[0]
    
    # Loop over the data to create overlapping sequences
    for i in range(num_samples - seq_length + 1):
        X_seq.append(data_tensor[i : i + seq_length])
        # For many-to-one classification, assign the label at the last time step of the sequence.
        y_seq.append(labels_tensor[i + seq_length - 1])
    
    # Stack the list of tensors into one tensor for sequences and labels
    X_seq = torch.stack(X_seq)
    y_seq = torch.stack(y_seq)
    
    return X_seq, y_seq

# ------------------------------
# 4. Example Usage for Preprocessing and Sequence Creation
# ------------------------------

# Specify the file path to your sensor data CSV file
file_path = "/Users/Garry/Desktop/HackaFuture/pump_sensor.csv"

# Load and preprocess data
data_tensor, labels_tensor, scaler, feature_cols = load_and_preprocess_data(file_path)
print("Processed data tensor shape:", data_tensor.shape)
if labels_tensor is not None:
    print("Labels tensor shape:", labels_tensor.shape)

# Define the sequence length (number of time steps per sequence)
seq_length = 60  # e.g., using 60 minutes per sequence

# Create sequences for LSTM classification
X_seq, y_seq = create_lstm_sequences(data_tensor, labels_tensor, seq_length)
print("LSTM input sequences shape (X_seq):", X_seq.shape)
print("LSTM labels shape (y_seq):", y_seq.shape)


# Assume X_seq and y_seq are produced from your preprocessing cell (both on GPU)
# Convert to NumPy arrays for splitting:
X_seq_np = X_seq.cpu().numpy()
y_seq_np = y_seq.cpu().numpy()

# Create an array of indices and split into training and validation indices (80/20 split)
indices = np.arange(len(X_seq_np))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

# Build training and validation datasets directly from the NumPy arrays:
train_dataset = TensorDataset(
    torch.tensor(X_seq_np[train_idx], dtype=torch.float32).to(device),
    torch.tensor(y_seq_np[train_idx], dtype=torch.long).to(device)
)
val_dataset = TensorDataset(
    torch.tensor(X_seq_np[val_idx], dtype=torch.float32).to(device),
    torch.tensor(y_seq_np[val_idx], dtype=torch.long).to(device)
)

# Create DataLoaders for batching:
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))

