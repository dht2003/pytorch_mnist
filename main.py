from mnist_loader import load
from mnist_visualize import visualize_data
import matplotlib.pyplot as plt
import model
import torch

n_epochs = 3


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load()
    network = model.Network()
    network.to(device)
    optimizer = model.create_opt(network)
    model.test(network, test_loader, device)
    for epoch in range(1, n_epochs + 1):
        model.train(network, optimizer, train_loader, epoch, device)
        model.test(network, test_loader, device)
    model.save(network, optimizer)


if __name__ == '__main__':
    main()
