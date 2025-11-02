#!/usr/bin/env python3
"""
Ultra-Simple Colab Runner
"""

# Install packages
!pip install pyyaml matplotlib torch torchvision -q

print("🚀 STARTING AUTO-CHECKPOINT TRAINING...")

# Import everything
from chckpnt_engine import enable_checkpointing, update_training_state
from Imges_AutoChckpnt import MultiTaskCNN, load_cifar10
import torch.optim as optim
import torch
import torch.nn as nn
import time

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MultiTaskCNN(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
trainloader = load_cifar10(batch_size=64)

# Start checkpointing
checkpointer = enable_checkpointing("config.yaml")
checkpointer.register_models({'model': (model, optimizer)})
tracker = checkpointer.start()

print("🎯 Starting Training with Checkpoint Updates...")

# Direct training loop with checkpoint updates
criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

model.train()
for epoch in range(1, 6):  # 5 epochs for testing
    running_loss = 0.0
    running_total = 0
    
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # Training step
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        class_output, reg_output = model(inputs)
        loss_cls = criterion_cls(class_output, labels)
        fake_target = torch.randn(labels.size(0), 1).to(device)
        loss_reg = criterion_reg(reg_output, fake_target)
        loss = loss_cls + 0.1 * loss_reg

        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = class_output.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)
        
        # Update running statistics
        running_loss = 0.95 * running_loss + 0.05 * loss.item() if running_total > 0 else loss.item()
        running_total += labels.size(0)

        # 🔄 UPDATE CHECKPOINT SYSTEM - THIS IS WHAT FIXES THE ISSUE
        update_training_state(epoch, batch_idx, running_loss, accuracy)

        # Progress reporting
        if batch_idx % 20 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(trainloader)}: '
                  f'Loss={running_loss:.4f}, Acc={accuracy:.2f}%')

        time.sleep(0.001)

    print(f"✓ Completed epoch {epoch}")

checkpointer.stop()
print("✅ TRAINING COMPLETED! Checkpoints now have REAL data!")
