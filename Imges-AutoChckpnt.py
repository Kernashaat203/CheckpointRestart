import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import yaml

# 💥 THE ESSENTIAL IMPORT FROM THE HOOK FILE
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
kind = chk_cfg['kind']
save_dir = chk_cfg['save_dir']


# --- MODEL DEFINITIONS (Global Access) ---
class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiTaskCNN, self).__init__()
        # Simple layers to simulate the original structure
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Linear(64 * 4 * 4, num_classes))
        self.regressor = nn.Sequential(nn.Linear(64 * 4 * 4, 1))

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


# --- PURE TRAINING FUNCTIONS ---
def load_cifar10():
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader


def train_pure_loop(model_name, model, optimizer, trainloader, criterion_cls, criterion_reg, device, tracker):
    """
    Main training function. Contains NO checkpointing or time-monitoring logic.
    It only updates the StateTracker.
    """
    total_batches = len(trainloader)
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    print(f"\n--- Training {model_name} (Pure Loop) ---")

    for epoch in range(max_epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(trainloader):

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

            # 🌟 The ONLY interaction with the checkpoint system: update the shared state
            tracker.update_state(epoch + 1, batch_idx, running_loss, accuracy)

            if batch_idx % 10 == 0:
                print(
                    f'[TRAIN] Epoch {epoch + 1}, Batch {batch_idx}/{total_batches}: Loss={running_loss:.4f}, Acc: {accuracy:.2f}%')

            time.sleep(0.001)  # Simulate training time

        print(f"✓ Completed epoch {epoch + 1}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader = load_cifar10()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # --- 🌟 THE MINIMAL CODE ADDITION (Hook Initialization) 🌟 ---

    manager = UniversalCheckpointManager()
    tracker = StateTracker()

    # Prepare data for the hook (using only the classification model for this run)
    hook_data = {
        'classification_model': (models['classification_model'].to(device),
                                 optimizers['classification_model'],
                                 tracker)
    }

    # Launch the autonomous timer thread
    timer = CheckpointTimer(
        interval=checkpoint_interval,
        kind=kind,
        save_dir=save_dir,
        models_optimizers=hook_data,
        manager=manager
    )
    timer.start()

    # --- END HOOK INITIALIZATION ---

    try:
        print("\n=========================================")
        print(f"AUTONOMOUS CHECKPOINTING SYSTEM STARTED")
        print("=========================================")

        # Run the pure training loop
        train_pure_loop('classification_model',
                        models['classification_model'],
                        optimizers['classification_model'],
                        trainloader,
                        criterion_cls,
                        criterion_reg,
                        device,
                        tracker)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping timer...")
        timer.stop()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
