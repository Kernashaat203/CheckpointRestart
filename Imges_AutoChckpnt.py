import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

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


def train_model(model, optimizer, trainloader, device, max_epochs=20):
    """
    Pure training loop - contains only training logic
    No checkpointing code here!
    """
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    print(f"▶ Starting training from beginning")

    total_batches = len(trainloader)
    running_loss = 0.0
    running_total = 0

    print(f"\n--- Training Model ---")
    
    model.train()
    
    for epoch in range(1, max_epochs + 1):
        for batch_idx, (inputs, labels) in enumerate(trainloader):
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

            # Progress reporting
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{total_batches}: '
                      f'Loss={running_loss:.4f}, Acc={accuracy:.2f}%')

            time.sleep(0.001)  # Simulate training time

        print(f"✓ Completed epoch {epoch}")


# Main Execution - Pure training setup
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Pure model setup
    model = MultiTaskCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Pure data loading
    trainloader = load_cifar10(batch_size=64)

    # Pure training execution
    print(f"\n🎯 Starting training...")
    train_model(
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        device=device,
        max_epochs=20
    )

    print("\n✅ Training completed successfully!")
