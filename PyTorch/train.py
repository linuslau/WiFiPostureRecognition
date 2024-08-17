import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.onnx
import netron
from dataset import getData
from datetime import datetime
from model import RNN

# Global constants
MAX_EPOCHS = 30  # Maximum number of epochs

def draw_train(train_acc, validation_acc, train_loss, validation_loss):
    print(train_acc)
    print(type(train_acc))
    # Save the Accuracy curve
    plt.subplot(2, 1, 1)
    plt.plot(train_acc)
    plt.plot(validation_acc)
    plt.xlabel("n_epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train_acc", "validation_acc"], loc=4)
    plt.ylim([0, 1])

    # Save the Loss curve
    plt.subplot(2, 1, 2)
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.xlabel("n_epoch")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "validation_loss"], loc=1)
    plt.ylim([0, 2])
    plt.show()

def getacc(target, label):
    # Use argmax on target and label to convert one-hot encoding to class indices
    _, predY = target.max(dim=1)
    _, trueY = label.max(dim=1)

    diff_count = torch.count_nonzero(predY - trueY)
    total_count = len(trueY)
    same_count = total_count - diff_count
    acc = same_count / total_count

    return acc, same_count, total_count

def main():
    # Initialization
    print("Initializing parameters")
    train_loss = []
    train_acc = []
    validation_loss = []
    validation_acc = []

    # Parameters
    learning_rate = 0.0001
    batch_size = 200
    display_step = 5

    # Network Parameters
    n_steps = 500  # timesteps, window_size
    input_size = 90  # WiFi activity data input (img shape: 90*window_size)
    hidden_size = 200  # hidden layer num of features original 200
    n_classes = 8  # WiFi activity total classes
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.device_count())  # Print the number of available GPUs
        print(torch.cuda.get_device_name(0))  # Print the name of the GPU
    # Check if GPU is used
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    model = RNN(input_size, hidden_size, n_classes).to(device)

    train_dataset, test_dataset = getData()

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n --Data download finished v1---")

    # DataLoader iteration for training data
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))  # Backpropagation

    start_time = datetime.now()
    bestacc = 0.0
    total_correct = 0.0
    total_count = 0.0

    for epoch in range(MAX_EPOCHS):
        # Training mode
        model.train()
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS} - Training:")
        for batchindex, (X, label) in enumerate(train_loader):
            Y = label.to(device).long()  # Keep in one-hot encoding form
            X = torch.transpose(X, 0, 1)
            X = X.to(device)

            predY = model(X)
            loss_train = criterion(predY, Y.argmax(dim=1))  # Compute loss using class indices

            # Backpropagation
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()  # Update gradients
            acc_train, same_count, total_count = getacc(predY, Y)

            if batchindex % display_step == 0:
                print(f'Batch {batchindex}, Loss: {loss_train:.3f}, Accuracy: {acc_train:.3f}')

        train_loss.append(loss_train.item())
        train_acc.append(acc_train.item())

        # Evaluation mode
        print("\nValidation:")

        model.eval()
        acc_val = 0
        total_val = 0
        total_correct = 0
        with torch.no_grad():
            for batchindex, (X, label) in enumerate(test_loader):
                Y = label.to(device).long()  # Keep in one-hot encoding form
                X = torch.transpose(X, 0, 1)
                X = X.to(device)
                predY = model(X)

                loss_val = criterion(predY, Y.argmax(dim=1))  # Compute loss using class indices
                _, same_count, total_count = getacc(predY, Y)
                total_correct += same_count
                total_val += total_count

            acc_val = total_correct / total_val
            validation_acc.append(acc_val.item())
            validation_loss.append(loss_val.item())

            if acc_val > bestacc:
                bestacc = acc_val
                torch.save(model, 'model.pt')

            print(f'Validation: Epoch: {epoch+1}, Validation Accuracy: {acc_val:.3f}, Total Samples: {total_count}')

        time_interval = datetime.now() - start_time
        print(f"Epoch {epoch+1} finished, Time: {time_interval.seconds:.2f} seconds")

    print(f"\nTraining finished, Time elapsed: {time_interval.seconds:.2f} seconds, Best Accuracy: {bestacc:.3f}")

    draw_train(train_acc, validation_acc, train_loss, validation_loss)

if __name__ == "__main__":
    main()
