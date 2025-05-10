import torch
import torch.nn as nn
from copy import deepcopy
from collections import defaultdict

def get_class_wise_lp_acc(model, train_loader, test_loader, criterion, device, num_classes, num_epochs=10, lr=0.001):
    # Clone the original model to avoid side effects
    lp_model = deepcopy(model)
    
    # Freeze all layers except the final fully connected layer
    for name, param in lp_model.named_parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    lp_model.fc = nn.Linear(lp_model.fc.in_features, num_classes)
    lp_model.eval()
    
    # Print layer trainability for verification
    for name, param in lp_model.named_parameters():
        print(f"{name} - requires_grad: {param.requires_grad}")
    
    # Move model to the specified device
    lp_model = lp_model.to(device)
    optimizer = torch.optim.Adam(lp_model.fc.parameters(), lr=lr)
    
    # Train the last layer only
    lp_model.fc.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = lp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Validate the model on the test set (class-wise accuracy)
    lp_model.eval()
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = lp_model(inputs)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(targets, predicted):
                total_per_class[t.item()] += 1
                if t == p:
                    correct_per_class[t.item()] += 1
    
    class_wise_accuracy = {cls: correct_per_class[cls] / total_per_class[cls] for cls in total_per_class}
    return class_wise_accuracy
