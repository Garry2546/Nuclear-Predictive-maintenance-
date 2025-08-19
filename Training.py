import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Data_prep import *

class PredictiveMaintenanceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5, bidirectional=True):
        """
        A robust LSTM-based classifier with attention.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension of the LSTM.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
            bidirectional (bool): Whether to use a bidirectional LSTM.
        """
        super(PredictiveMaintenanceLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism: compute attention weights per time step
        self.attention = nn.Linear(lstm_output_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_length, lstm_output_dim)
        
        # Compute attention weights across time steps
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_length, 1)
        # Compute context vector as weighted sum of LSTM outputs
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, lstm_output_dim)
        context = self.dropout(context)
        logits = self.fc(context)  # (batch, num_classes)
        return logits, attn_weights

# ------------------------------
# 6. Model Instantiation and Training Setup
# ------------------------------
input_dim = X_seq.shape[2]
num_classes = len(torch.unique(y_seq))
hidden_dim = 128
num_layers = 2
dropout = 0.5

model = PredictiveMaintenanceLSTM(input_dim, hidden_dim, num_layers, num_classes, dropout, bidirectional=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

num_epochs = 50
clip_value = 1.0
best_val_loss = float('inf')

# ------------------------------
# 7. Training Loop
# ------------------------------
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Apply gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        total_train += y_batch.size(0)
        correct_train += (preds == y_batch).sum().item()
    
    avg_train_loss = train_loss / total_train
    train_acc = correct_train / total_train
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            total_val += y_batch.size(0)
            correct_val += (preds == y_batch).sum().item()
    
    avg_val_loss = val_loss / total_val
    val_acc = correct_val / total_val
    
    scheduler.step(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save the best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_predictive_model.pth")
        print("Saved best model with val loss:", best_val_loss)

print("Training complete.")