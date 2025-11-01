from chckpnt_engine import AutonomousCheckpointer
from main import MultiTaskCNN, load_cifar10, train_model
import torch.optim as optim

# Setup models and data
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiTaskCNN(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
trainloader = load_cifar10(batch_size=64)

# Initialize checkpoint system
checkpointer = AutonomousCheckpointer("config.yaml")
checkpointer.register_models({
    'my_model': (model, optimizer)
})

# Start checkpointing and get tracker
tracker = checkpointer.start()

# Run training with checkpointing
train_model(model, optimizer, trainloader, device, tracker)

checkpointer.stop()
