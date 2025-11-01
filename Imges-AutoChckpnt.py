import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import yaml

# Import the autonomous checkpointing system
from chckpnt_engine import AutonomousCheckpointer

# Load user configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Pure Model Definition (No checkpointing logic)
class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiTaskCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.regressor = nn.Sequential(
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        classification_output = self.classifier(x)
        regression_output = self.regressor(x)
        return classification_output, regression_output


# Pure Training Functions (No checkpointing logic)
def load_cifar10(batch_size):
    """Load dataset - pure data loading"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                            shuffle=True, num_workers=2)
    return trainloader


def calculate_accuracy(outputs, labels):
    """Calculate accuracy - pure metric calculation"""
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    return 100.0 * correct / labels.size(0)


def train_model(model, optimizer, trainloader, device, tracker, model_name):
    """
    Pure training loop - contains only training logic
    Checkpointing is handled automatically by the system
    """
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    # Check if we're resuming from checkpoint
    checkpoint_loaded = tracker.get_state()[0] > 0
    start_epoch = tracker.get_state()[0] if checkpoint_loaded else 1
    start_batch = tracker.get_state()[1] if checkpoint_loaded else 0
    
    if checkpoint_loaded:
        print(f"↻ Resuming {model_name} from epoch {start_epoch}, batch {start_batch}")
    else:
        print(f"▶ Starting {model_name} from beginning")

    total_batches = len(trainloader)
    running_loss = tracker.get_state()[2] if checkpoint_loaded else 0.0
    running_total = 0

    print(f"\n--- Training {model_name} ---")
    
    try:
        model.train()
        
        for epoch in range(start_epoch, config['training']['max_epochs'] + 1):
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                # Skip already processed batches when resuming
                if batch_idx < start_batch:
                    continue

                # Check if checkpoint system is still running
                if not any(timer.is_running for timer in tracker.timers):
                    print("⏹️ Session completed - stopping training")
                    return

                # Pure training step
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                class_output, reg_output = model(inputs)
                loss_cls = criterion_cls(class_output, labels)
                fake_target = torch.randn(labels.size(0), 1).to(device)
                loss_reg = criterion_reg(reg_output, fake_target)
                loss = loss_cls + 0.1 * loss_reg

                loss.backward()
                optimizer.step()

                # Update running statistics
                running_loss = 0.95 * running_loss + 0.05 * loss.item() if running_total > 0 else loss.item()
                accuracy = calculate_accuracy(class_output, labels)
                running_total += labels.size(0)

                # 🔄 ONLY INTERACTION WITH CHECKPOINT SYSTEM: update state
                tracker.update_state(epoch, batch_idx, running_loss, accuracy)
                
                # Collect snapshot data if enabled
                if (config['checkpoint']['enable_snapshots'] and 
                    batch_idx % config['checkpoint']['snapshot_interval'] == 0):
                    tracker.add_snapshot_data(inputs[0], class_output[0], labels[0])

                # Progress reporting
                if batch_idx % 20 == 0:
                    _, _, _, _, session_start, total_elapsed = tracker.get_state()
                    current_time = time.time() - session_start
                    print(f'Epoch {epoch}, Batch {batch_idx}/{total_batches}: '
                          f'Loss={running_loss:.4f}, Acc={accuracy:.2f}%, '
                          f'Time={current_time/60:.1f}m')

                time.sleep(0.001)  # Simulate training time

            # Reset for next epoch
            start_batch = 0
            print(f"✓ Completed epoch {epoch}")

    except Exception as e:
        print(f"✗ Training error: {e}")
        import traceback
        traceback.print_exc()


# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Pure model setup
    models = {
        'classification_model': MultiTaskCNN(num_classes=config['training']['num_classes']).to(device),
        'feature_model': MultiTaskCNN(num_classes=config['training']['num_classes']).to(device)
    }

    optimizers = {
        name: optim.Adam(model.parameters(), 
                        lr=config['training']['learning_rate'],
                        weight_decay=config['training']['weight_decay'])
        for name, model in models.items()
    }

    # Pure data loading
    trainloader = load_cifar10(config['training']['batch_size'])

    # 🚀 Initialize autonomous checkpointing system (ONE-TIME SETUP)
    checkpointer = AutonomousCheckpointer(
        config=config,
        models_optimizers={
            name: (model, optimizer, None)  # Tracker will be added by system
            for name, (model, optimizer) in zip(models.keys(), zip(models.values(), optimizers.values()))
        }
    )
    
    # Start the system and get the state tracker
    tracker = checkpointer.start()

    try:
        # Pure training execution
        for model_name, model in models.items():
            print(f"\n🎯 Training {model_name}...")
            train_model(
                model=model,
                optimizer=optimizers[model_name],
                trainloader=trainloader,
                device=device,
                tracker=tracker,
                model_name=model_name
            )

        print("\n✅ All models trained successfully!")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        checkpointer.stop()
