import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models.cnn_cifar10.cnn import CNN

BATCH_SIZE = 4
CALIB_TEST_SPLIT = 0.8


def distance(_outputs: np.ndarray, _train_outputs: np.ndarray):
    return np.min(
        np.mean((_outputs[None] - _train_outputs[:, None]) ** 2, axis=-1), axis=0
    )


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar10_train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    cifar10_train_data_loader = torch.utils.data.DataLoader(
        cifar10_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    cifar10_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    cifar10_data_loader = torch.utils.data.DataLoader(
        cifar10_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    cifar100_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    cifar100_data_loader = torch.utils.data.DataLoader(
        cifar100_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    cnn = CNN()
    cnn.load_state_dict(torch.load("../../../models/cnn_cifar10/params.pth"))
    cnn.eval()

    with torch.no_grad():
        train_outputs = []
        for inputs, _ in cifar10_train_data_loader:
            train_outputs.append(cnn(inputs))
        train_outputs = torch.cat(train_outputs).numpy()

        outputs, probs, targets = [], [], []
        for inputs, _targets in cifar10_data_loader:
            outputs.append(cnn(inputs))
            probs.append(F.softmax(outputs[-1], dim=-1))
            targets.append(_targets)
        outputs = torch.cat(outputs).numpy()
        probs = torch.cat(probs).numpy()
        targets = torch.cat(targets).numpy()

    distances = distance(outputs, train_outputs)

    np.save(
        "./cifar10_data.npy",
        dict(
            probs=probs,
            distances=distances,
            targets=targets,
        ),
    )

    with torch.no_grad():
        outputs, probs, targets = [], [], []
        for inputs, _targets in cifar100_data_loader:
            outputs.append(cnn(inputs))
            probs.append(F.softmax(outputs[-1], dim=-1))
            targets.append(_targets)
        outputs = torch.cat(outputs).numpy()
        probs = torch.cat(probs).numpy()
        targets = torch.cat(targets).numpy()

    distances = distance(outputs, train_outputs)

    np.save(
        "./cifar100_data.npy",
        dict(
            probs=probs,
            distances=distances,
            targets=targets,
        ),
    )
