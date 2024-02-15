import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from models.cnn_cifar10.cnn import CNN

BATCH_SIZE = 64
PRINT_EVERY_N_ITERS = 1
N_EPOCHS = 30

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    cnn = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    strformat = "Epoch: {:<10} | Iter: {:<10} | Loss: {:<20}"
    for epoch in range(N_EPOCHS):
        running_loss = 0.
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = cnn(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % PRINT_EVERY_N_ITERS == PRINT_EVERY_N_ITERS - 1:
                print(strformat.format(epoch + 1, i + 1, running_loss / PRINT_EVERY_N_ITERS))
                running_loss = 0.

    print('Finished training.')

    PATH = './params.pth'
    torch.save(cnn.state_dict(), PATH)
