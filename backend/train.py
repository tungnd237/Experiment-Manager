import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sqlalchemy.orm import Session
from models import Experiment

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def evaluate_model(model, test_loader, loss_fn):
    """ Compute accuracy and validation loss on the test dataset """
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    batch_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            batch_count += 1
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total, val_loss / batch_count  # Accuracy & Avg Validation Loss

def train_experiment(db: Session, experiment_id: int, lr: float, batch_size: int, epochs: int):
    print(f"🚀 Training started for Experiment {experiment_id}...")

    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, transform=transform, download=True),
        batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False, transform=transform, download=True),
        batch_size=1000, shuffle=False
    )

    model = MNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} running...")
        model.train()
        epoch_loss = 0
        batch_count = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count

        accuracy, avg_val_loss = evaluate_model(model, test_loader, loss_fn)

        db.query(Experiment).filter(Experiment.id == experiment_id).update({
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss, 
            "epoch": epoch + 1,  
            "accuracy": accuracy
        })
        db.commit()

    print(f"Training completed for Experiment {experiment_id}, Accuracy: {accuracy:.4f}")
    db.query(Experiment).filter(Experiment.id == experiment_id).update({"status": "completed"})
    db.commit()
