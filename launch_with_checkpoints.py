#!/usr/bin/env python3
"""
Runner Script - Integrates checkpointing with pure ML model
User doesn't need to modify their pure ML code!
"""

import torch
import torch.optim as optim
import torch.nn as nn
import time
# Import user's pure ML code (NO CHANGES NEEDED)
from Imges_AutoChckpnt import MultiTaskCNN, load_cifar10, calculate_accuracy

# Import checkpoint engine (separate file)
from chckpnt_engine import enable_checkpointing, update_training_state, add_snapshot_data, should_stop_training


def create_config():
    """Create configuration file"""
    config_content = """# User Configuration File
checkpoint:
  checkpoint_interval: 300   
  max_session_time: 900       
  save_dir: "./checkpoints"
  enable_snapshots: true

training:
  batch_size: 64
  learning_rate: 0.001
  max_epochs: 20
"""
    with open("config.yaml", "w") as f:
        f.write(config_content)
    print("✓ Created config.yaml")


def train_with_checkpoints():
    """Enhanced training function with checkpointing"""
    print("🚀 STARTING TRAINING WITH AUTO-CHECKPOINTS...")
    print("==============================================")

    # Setup (same as user's original code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model and optimizer (same as user's original code)
    model = MultiTaskCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Load data (same as user's original code)
    trainloader = load_cifar10(batch_size=64)
    total_batches = len(trainloader)

    # Start checkpointing system
    checkpointer = enable_checkpointing("config.yaml")
    checkpointer.register_models({'main_model': (model, optimizer)})
    tracker = checkpointer.start()

    print("🎯 Starting Training with Time-Based Checkpoints...")
    print("   - Checkpoints every 5 minutes")
    print("   - Auto-stop at 15 minutes")
    print("   - Resume from exact batch when restarted")
    print("==============================================")

    # Load checkpoint if exists
    checkpoint_data = checkpointer.manager.load_checkpoint(model, optimizer, 'main_model')

    if checkpoint_data:
        start_epoch = checkpoint_data['epoch']
        start_batch = checkpoint_data['batch_idx'] + 1  # Resume from NEXT batch
        running_loss = checkpoint_data.get('loss', 0.0) or 0.0
        print(f"✓ Resuming from epoch {start_epoch}, batch {start_batch}")
    else:
        start_epoch = 1
        start_batch = 0
        running_loss = 0.0
        print("✓ Starting from beginning")

    # Training loop with checkpoint integration
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    model.train()
    running_correct = 0
    running_total = 0

    try:
        for epoch in range(start_epoch, 1000):  # Large number for time-based stopping
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                # Skip batches we've already processed
                if batch_idx < start_batch:
                    continue

                # Check if we should stop due to time limit
                if should_stop_training():
                    print("🛑 Time limit reached - stopping training")
                    checkpointer.stop()
                    return

                # Training step (same as user's original code)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                class_output, reg_output = model(inputs)
                loss_cls = criterion_cls(class_output, labels)
                fake_target = torch.randn(labels.size(0), 1).to(device)
                loss_reg = criterion_reg(reg_output, fake_target)
                loss = loss_cls + 0.1 * loss_reg

                loss.backward()
                optimizer.step()

                # Calculate accuracy (same as user's original code)
                accuracy = calculate_accuracy(class_output, labels)

                # Update running statistics
                running_total += labels.size(0)
                running_correct += (accuracy * labels.size(0) / 100)  # Convert back to count
                running_loss = 0.95 * running_loss + 0.05 * loss.item() if running_total > 0 else loss.item()

                # 🔄 ONLY ADDITION: Update checkpoint system
                update_training_state(epoch, batch_idx, running_loss, accuracy)

                # Optional: Add snapshot data
                if batch_idx % 100 == 0:
                    add_snapshot_data(inputs[0], class_output[0], labels[0])

                # Progress reporting (similar to user's original code)
                if batch_idx % 20 == 0:
                    epoch_accuracy = 100.0 * running_correct / running_total if running_total > 0 else 0.0
                    print(f'Epoch {epoch}, Batch {batch_idx}/{total_batches}: '
                          f'Loss={running_loss:.4f}, Acc={epoch_accuracy:.2f}%')

                time.sleep(0.001)  # Simulate training time

            # Reset for next epoch
            start_batch = 0
            running_correct = 0
            running_total = 0
            print(f"✓ Completed epoch {epoch}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        checkpointer.stop()

    print("✅ TRAINING COMPLETED!")


if __name__ == "__main__":
    # Create config file if it doesn't exist
    try:
        with open("config.yaml", "r") as f:
            pass
    except FileNotFoundError:
        create_config()

    train_with_checkpoints()
