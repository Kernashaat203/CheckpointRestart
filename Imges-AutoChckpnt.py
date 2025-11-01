import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import yaml
import os

# Enhanced imports from the hook file
from chckpnt_engine import UniversalCheckpointManager, CheckpointTimer, StateTracker

# --- CONFIG LOADING & GLOBALS ---
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

chk_cfg = cfg['checkpoint']
train_cfg = cfg['training']

batch_size = train_cfg['batch_size']
learning_rate = train_cfg['learning_rate']
weight_decay = train_cfg['weight_decay']
max_epochs = train_cfg['max_epochs']
checkpoint_interval = chk_cfg['checkpoint_interval']
max_session_time = chk_cfg['max_session_time']
kind = chk_cfg['kind']
save_dir = chk_cfg['save_dir']


# --- ENHANCED MODEL DEFINITION ---
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


# Global Model/Optimizer Instances
model_cls = MultiTaskCNN(num_classes=10)
model_feat = MultiTaskCNN(num_classes=10)

models = {
    'classification_model': model_cls,
    'feature_model': model_feat
}
optimizers = {
    name: optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for name, model in models.items()
}


# --- ENHANCED TRAINING FUNCTIONS ---
def load_cifar10():
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader


def train_with_resumption(model_name, model, optimizer, trainloader, criterion_cls, criterion_reg, device, tracker, manager):
    """
    Enhanced training function with checkpoint resumption and snapshot collection
    """
    # Load checkpoint if exists
    checkpoint_data = manager.load_checkpoint(model, optimizer, model_name)
    
    if checkpoint_data:
        start_epoch = checkpoint_data['epoch']
        start_batch = checkpoint_data['batch_idx'] + 1  # Start from next batch
        running_loss = checkpoint_data.get('loss', 0.0) or 0.0
        running_correct = 0
        running_total = 0
        print(f"✓ Resuming {model_name} from epoch {start_epoch}, batch {start_batch}")
    else:
        start_epoch = 1
        start_batch = 0
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        print(f"✓ Starting {model_name} from beginning")

    total_batches = len(trainloader)
    
    print(f"\n--- Training {model_name} (With Time-Based Checkpoints) ---")

    try:
        model.train()
        
        # Continue training from where we left off
        for epoch in range(start_epoch, max_epochs + 1):
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                # Skip batches we've already processed
                if batch_idx < start_batch:
                    continue

                # Check if timer is still running
                if not any(timer.is_running for timer in tracker.timers) if hasattr(tracker, 'timers') else False:
                    print(f"[TRAIN] Session time limit reached, stopping training...")
                    return

                # --- Training Logic ---
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                class_output, reg_output = model(inputs)

                loss_cls = criterion_cls(class_output, labels)
                fake_target = torch.randn(labels.size(0), 1).to(device)
                loss_reg = criterion_reg(reg_output, fake_target)
                loss = loss_cls + 0.1 * loss_reg

                loss.backward()
                optimizer.step()
                # --- End Training Logic ---

                # Update running statistics
                if running_total == 0:
                    running_loss = loss.item()
                else:
                    running_loss = 0.95 * running_loss + 0.05 * loss.item()

                _, predicted = class_output.max(1)
                running_total += labels.size(0)
                running_correct += predicted.eq(labels).sum().item()
                accuracy = 100.0 * running_correct / running_total if running_total > 0 else 0.0

                # Update the shared state
                tracker.update_state(epoch, batch_idx, running_loss, accuracy)
                
                # Collect snapshot data every 100 batches
                if batch_idx % 100 == 0:
                    tracker.add_snapshot_data(inputs[0], class_output[0], labels[0])

                # Print progress
                if batch_idx % 10 == 0:
                    _, _, _, _, session_start, total_elapsed = tracker.get_state()
                    current_time = time.time() - session_start
                    elapsed_minutes = total_elapsed / 60
                    
                    print(f'[TRAIN] Epoch {epoch}, Batch {batch_idx}/{total_batches}: '
                          f'Loss={running_loss:.4f}, Acc: {accuracy:.2f}%, '
                          f'Time: {current_time/60:.1f}m (+{elapsed_minutes:.1f}m total)')

                time.sleep(0.001)  # Simulate training time

            # Reset for next epoch
            start_batch = 0
            print(f"✓ Completed epoch {epoch} for {model_name}")

    except Exception as e:
        print(f"[TRAIN] ✗ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader = load_cifar10()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # --- ENHANCED HOOK INITIALIZATION ---
    manager = UniversalCheckpointManager(save_dir=save_dir)
    tracker = StateTracker()

    # Prepare data for the hook
    hook_data = {
        'classification_model': (models['classification_model'].to(device),
                                 optimizers['classification_model'],
                                 tracker),
        'feature_model': (models['feature_model'].to(device),
                          optimizers['feature_model'],
                          tracker)
    }

    # Launch the enhanced autonomous timer thread
    timer = CheckpointTimer(
        interval=checkpoint_interval,
        kind=kind,
        save_dir=save_dir,
        models_optimizers=hook_data,
        manager=manager,
        max_session_time=max_session_time
    )

    print(f"\n{'='*60}")
    print(f"ENHANCED TIME-BASED CHECKPOINTING SYSTEM")
    print(f"Maximum session time: {max_session_time/60} minutes")
    print(f"Checkpoint interval: {checkpoint_interval} seconds")
    print(f"{'='*60}")

    try:
        timer.start()

        print("\n=========================================")
        print(f"AUTONOMOUS CHECKPOINTING SYSTEM STARTED")
        print("=========================================")

        # Run training for each model
        for model_name in ['classification_model', 'feature_model']:
            print(f"\n>>> Starting training for: {model_name}")
            train_with_resumption(
                model_name,
                models[model_name],
                optimizers[model_name],
                trainloader,
                criterion_cls,
                criterion_reg,
                device,
                tracker,
                manager
            )

        # Session complete
        _, _, _, _, session_start, total_elapsed = tracker.get_state()
        total_session_time = time.time() - session_start
        total_training_time = total_elapsed + total_session_time

        print(f"\n{'='*60}")
        print(f"SESSION COMPLETE")
        print(f"Session time: {total_session_time/60:.1f} minutes")
        print(f"Total training time: {total_training_time/60:.1f} minutes")
        print(f"Checkpoints saved for resumption")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping timer...")
        timer.stop()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        timer.stop()
