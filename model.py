import pandas as pd
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import itertools
import json
import os

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda')

df = pd.read_csv('./kc_house_data.csv')

df['date_timestamp'] = pd.to_datetime(df['date']).astype(int) / 10**9
df['date_timestamp'] = df['date_timestamp'] / (24 * 3600)
df['floors'] = df['floors'].astype(float)

numerical_feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                          'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                          'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                          'sqft_living15', 'sqft_lot15', 'date_timestamp']

X_numerical = df[numerical_feature_cols].values
X_zipcode = df[['zipcode']]
y = df['price'].values.reshape(-1, 1)

X_num_train, X_num_test, X_zip_train, X_zip_test, y_train, y_test = train_test_split(
    X_numerical, X_zipcode, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_num_train_scaled = scaler_X.fit_transform(X_num_train)
X_num_test_scaled = scaler_X.transform(X_num_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_zip_train_df = pd.DataFrame(X_zip_train, columns=['zipcode'])
X_zip_test_df = pd.DataFrame(X_zip_test, columns=['zipcode'])

X_zip_train_encoded = pd.get_dummies(X_zip_train_df, columns=['zipcode'], prefix='zip')

X_zip_test_encoded = pd.get_dummies(X_zip_test_df, columns=['zipcode'], prefix='zip')

X_zip_test_encoded = X_zip_test_encoded.reindex(columns=X_zip_train_encoded.columns, fill_value=0)

X_zip_train_encoded = X_zip_train_encoded.values
X_zip_test_encoded = X_zip_test_encoded.values

X_train_scaled = np.concatenate([X_num_train_scaled, X_zip_train_encoded], axis=1)
X_test_scaled = np.concatenate([X_num_test_scaled, X_zip_test_encoded], axis=1)

X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

print(f"Training samples: {len(X_train_scaled)}")
print(f"Test samples: {len(X_test_scaled)}")
print(f"Number of numerical features: {X_num_train_scaled.shape[1]}")
print(f"Number of zipcode features (one-hot): {X_zip_train_encoded.shape[1]}")
print(f"Total number of features: {X_train_scaled.shape[1]}")
print(f"Data normalized with StandardScaler\n")

class HousePriceNN(nn.Module):
    def __init__(self, input_dim, hidden_width, num_hidden_layers, activation_fn, dropout_rate=0.0):
        super(HousePriceNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_width))
        layers.append(activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_model(model, X_train, y_train, X_test, y_test, lr, batch_size, weight_decay, epochs=10, iteration_num=0):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    epoch_losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test)
        test_rmse = torch.sqrt(test_loss)
    
    return test_rmse.item(), epoch_losses

hyperparameter_space = {
    'hidden_width': [32, 64, 128, 256],
    'num_hidden_layers': [2, 3, 4],
    'activation_fn': [nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.ELU],
    'learning_rate': [0.001, 0.005, 0.01],
    'dropout_rate': [0.0, 0.1, 0.2],
    'batch_size': [128, 256],
    'weight_decay': [0.0, 0.01, 0.05, 0.1]
}

all_combinations = list(itertools.product(
    hyperparameter_space['hidden_width'],
    hyperparameter_space['num_hidden_layers'],
    hyperparameter_space['activation_fn'],
    hyperparameter_space['learning_rate'],
    hyperparameter_space['dropout_rate'],
    hyperparameter_space['batch_size'],
    hyperparameter_space['weight_decay']
))

print(f"Total possible combinations: {len(all_combinations)}")

n_iterations = min(100, len(all_combinations))
sampled_combinations = random.sample(all_combinations, n_iterations)

input_dim = X_train_scaled.shape[1]

best_rmse = float('inf')
best_params = None

runs_file = 'runs.json'
if os.path.exists(runs_file):
    with open(runs_file, 'r') as f:
        all_runs = json.load(f)
else:
    all_runs = []

print(f"Sampling {n_iterations} random combinations for hyperparameter search...\n")

for i, combination in enumerate(sampled_combinations):
    hidden_width, num_hidden_layers, activation_fn, learning_rate, dropout_rate, batch_size, weight_decay = combination
    
    params = {
        'hidden_width': hidden_width,
        'num_hidden_layers': num_hidden_layers,
        'activation_fn': activation_fn.__name__,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'weight_decay': weight_decay
    }
    
    print(f"Starting Trial {i+1}")
    
    model = HousePriceNN(
        input_dim=input_dim,
        hidden_width=hidden_width,
        num_hidden_layers=num_hidden_layers,
        activation_fn=activation_fn,
        dropout_rate=dropout_rate
    )
    
    test_rmse, epoch_losses = train_model(
        model, X_train_tensor, y_train_tensor, 
        X_test_tensor, y_test_tensor,
        lr=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=10,
        iteration_num=i+1
    )
    
    test_rmse_denorm = test_rmse * scaler_y.scale_[0]
    print(f"[Trial {i+1}] Test RMSE (actual): ${test_rmse_denorm:,.2f}")
    
    run_data = params.copy()
    run_data['test_rmse_normalized'] = test_rmse
    run_data['test_rmse'] = test_rmse_denorm
    run_data['epoch_losses'] = epoch_losses
    
    all_runs.append(run_data)
    
    with open(runs_file, 'w') as f:
        json.dump(all_runs, f, indent=2)
    
    if test_rmse < best_rmse:
        best_rmse = test_rmse
        best_rmse_denorm = test_rmse_denorm
        best_params = params.copy()
        best_params['activation_fn'] = hyperparameter_space['activation_fn'][
            [fn.__name__ for fn in hyperparameter_space['activation_fn']].index(params['activation_fn'])
        ]

print("\n" + "="*60)
print("BEST HYPERPARAMETERS FOUND:")
print("="*60)
print(f"Hidden width: {best_params['hidden_width']}")
print(f"Num hidden layers: {best_params['num_hidden_layers']}")
print(f"Activation: {best_params['activation_fn'].__name__}")
print(f"Learning rate: {best_params['learning_rate']}")
print(f"Dropout rate: {best_params['dropout_rate']}")
print(f"Batch size: {best_params['batch_size']}")
print(f"Weight decay: {best_params['weight_decay']}")
print(f"\nBest Test RMSE (normalized): {best_rmse:.6f}")
print(f"Best Test RMSE (actual): ${best_rmse_denorm:,.2f}")

print("\n" + "="*60)
print("Training final model with best parameters...")
print("="*60)

final_model = HousePriceNN(
    input_dim=input_dim,
    hidden_width=best_params['hidden_width'],
    num_hidden_layers=best_params['num_hidden_layers'],
    activation_fn=best_params['activation_fn'],
    dropout_rate=best_params['dropout_rate']
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

epochs = 100
best_test_rmse = float('inf')
best_model_state = None

final_model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = final_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    
    final_model.eval()
    with torch.no_grad():
        test_pred = final_model(X_test_tensor)
        test_loss = criterion(test_pred, y_test_tensor)
        test_rmse = torch.sqrt(test_loss).item()
    final_model.train()
    
    if test_rmse < best_test_rmse:
        best_test_rmse = test_rmse
        best_model_state = copy.deepcopy(final_model.state_dict())
        best_epoch = epoch + 1
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.6f}, Test RMSE: {test_rmse:.6f}")

final_model.load_state_dict(best_model_state)
print(f"\nLoaded best model from epoch {best_epoch}")

final_model.eval()
with torch.no_grad():
    train_pred = final_model(X_train_tensor)
    test_pred = final_model(X_test_tensor)
    
    train_rmse = torch.sqrt(criterion(train_pred, y_train_tensor))
    test_rmse = torch.sqrt(criterion(test_pred, y_test_tensor))
    
    train_mae = torch.mean(torch.abs(train_pred - y_train_tensor))
    test_mae = torch.mean(torch.abs(test_pred - y_test_tensor))
    
    train_r2 = 1 - torch.sum((y_train_tensor - train_pred)**2) / torch.sum((y_train_tensor - torch.mean(y_train_tensor))**2)
    test_r2 = 1 - torch.sum((y_test_tensor - test_pred)**2) / torch.sum((y_test_tensor - torch.mean(y_test_tensor))**2)
    
    train_pred_denorm = scaler_y.inverse_transform(train_pred.cpu().numpy())
    test_pred_denorm = scaler_y.inverse_transform(test_pred.cpu().numpy())

train_rmse_denorm = train_rmse.item() * scaler_y.scale_[0]
test_rmse_denorm = test_rmse.item() * scaler_y.scale_[0]
train_mae_denorm = train_mae.item() * scaler_y.scale_[0]
test_mae_denorm = test_mae.item() * scaler_y.scale_[0]

print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE:")
print("="*60)
print(f"Train RMSE (normalized): {train_rmse.item():.6f}")
print(f"Train RMSE (actual): ${train_rmse_denorm:,.2f}")
print(f"Train MAE (actual): ${train_mae_denorm:,.2f}")
print(f"Train R²: {train_r2.item():.6f}")
print(f"Test RMSE (normalized): {test_rmse.item():.6f}")
print(f"Test RMSE (actual): ${test_rmse_denorm:,.2f}")
print(f"Test MAE (actual): ${test_mae_denorm:,.2f}")
print(f"Test R²: {test_r2.item():.6f}")

print("\n" + "="*60)
print("SAMPLE PREDICTIONS:")
print("="*60)
sample_indices = random.sample(range(len(X_test_scaled)), 5)
for idx in sample_indices:
    actual = y_test[idx, 0]
    predicted = test_pred_denorm[idx, 0]
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100
    print(f"Actual: ${actual:,.2f} | Predicted: ${predicted:,.2f} | Error: ${error:,.2f} ({error_pct:.1f}%)")

print("\n" + "="*60)
print("GENERATING ACTUAL VS PREDICTED PLOT:")
print("="*60)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_pred_denorm, alpha=0.5, s=10, label='Predictions')

min_val = min(y_test.min(), test_pred_denorm.min())
max_val = max(y_test.max(), test_pred_denorm.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Price ($)', fontsize=12)
plt.ylabel('Predicted Price ($)', fontsize=12)
plt.title('Actual vs Predicted House Prices', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('square')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'actual_vs_predicted.png'")
