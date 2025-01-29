import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Define the MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Parameters
input_size = 784  # Example for MNIST dataset (28x28 images)
hidden_size = 32
num_classes = 10
num_epochs = 10
batch_size = 1
learning_rate = 0.001

# Dummy dataset (replace with actual data)
x_train = torch.randn(600, input_size)
y_train = torch.randint(0, num_classes, (600,))

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleMLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 100 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# # Save the trained model
torch.save(model.state_dict(), 'simple_mlp.pth')

print('Model training complete and saved to simple_mlp.pth')
