from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from forecaster.utils.data import get_loaders
from forecaster.utils.describers import Message
from forecaster.utils.describers import Signal
from forecaster.utils.io import load_one_bms


def train(data_loader, model, criterion, optimizer, device, debug=False):
    model.train()
    num_batches = len(data_loader)

    total_loss = 0
    for i, (X, y) in enumerate(data_loader):

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(X)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if debug:
            print(f"Progress: {100 * i / len(data_loader):.2f} % , Loss: {total_loss / (i + 1):.4f}")

    avg_loss = total_loss / num_batches

    return avg_loss


def evaluate(data_loader, model, criterion, device, mode='Val', debug=False):
    model.eval()
    num_batches = len(data_loader)

    preds = []
    labels = []
    total_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            output = model(X)
            total_loss += criterion(output, y).item()

            preds.extend([pred.item() for pred in output])
            labels.extend([label.item() for label in y])

            if debug:
                print(f"Progress: {i / len(data_loader):.2f}, Loss: {total_loss / (i + 1):.4f}")

    avg_loss = total_loss / num_batches

    if mode == 'Test':
        plt.figure()
        plt.plot(labels, label='Labels')
        plt.plot(preds, label=f'Predictions, loss={avg_loss:.2f}')
        plt.grid()
        plt.show()

    return avg_loss


def run(model, name, device, epochs=10, lr=0.001, bs=128, sequence_length=300, feature_first=False):
    data_dir = Path("../../data/bms1/")

    required_messages = [
        Message(
            name='batt_mod_temperatures_a',
            limits=(-20, 70),
            signals=[
                Signal(name='batt_t_module1')
            ]
        )
    ]

    data = load_one_bms(data_dir, required_messages, debug=True)

    train_loader, val_loader, test_loader = get_loaders(
        data,
        sequence_length=sequence_length,
        bs=bs,
        feature_first=feature_first
    )

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = np.inf  # evaluate(val_loader, model, criterion, device, debug=True, mode='Val')
    # print(f"Initial Validation Loss: {best_val_loss:.5f}")

    for epoch in range(epochs):
        train_loss = train(train_loader, model, criterion, optimizer=optimizer, device=device, debug=True)
        val_loss = evaluate(val_loader, model, criterion, device, mode='Val')

        info = f"Epoch {epoch}: Training Loss: {train_loss:.5f}\tValidation Loss: {val_loss:.5f}"

        if best_val_loss > val_loss:
            torch.save(model.state_dict(), name + '.net')
            best_val_loss = val_loss

            info += "\tCheckpoint!"

            test_loss = evaluate(test_loader, model, criterion, device, mode='Test')

        print(info)
