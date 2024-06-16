# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from ucf101_dataset import UCF101Dataset
from video_model import YourModel  # Replace YourModel with your actual model class

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        # Remove transforms.ToTensor() because read_video already returns a tensor
    ])

    # Create datasets
    train_dataset = UCF101Dataset(root_dir='UCF-101', annotation_file='UCF-101/ucfTrainTestlist/trainlist01.txt', transform=transform)
    val_dataset = UCF101Dataset(root_dir='UCF-101', annotation_file='UCF-101/ucfTrainTestlist/testlist01.txt', transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = YourModel(num_classes=101)  # Adjust based on your model's requirements
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(10):  # Change number of epochs as needed
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader)}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')

if __name__ == '__main__':
    main()

