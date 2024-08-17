import torch
from model import RNN
from dataset import getData
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 假设你的数据集标签名称如下
class_labels = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk", "unknown"]

def inference(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []
    with torch.no_grad():  # No need to compute gradients during inference
        for X, labels in data_loader:
            X = torch.transpose(X, 0, 1).to(device)  # Move data to the specified device (GPU or CPU)
            labels = labels.to(device)  # Move labels to the specified device
            output = model(X)
            pred = torch.argmax(output, dim=1)  # Get the predicted class indices
            predictions.extend(pred.cpu().numpy())  # Move predictions back to CPU and convert to numpy array
            true_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())  # Convert one-hot encoded labels to indices and move to CPU
    return predictions, true_labels

def plot_class_distribution(cm, class_labels):
    # Ensure that confusion matrix is square
    if cm.shape[0] != len(class_labels):
        print("Mismatch between class labels and confusion matrix size. Adjusting...")
        full_cm = np.zeros((len(class_labels), len(class_labels)))
        full_cm[:cm.shape[0], :cm.shape[1]] = cm
        cm = full_cm

    # Calculate accuracy per class
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    
    # Plot accuracy per class with class labels
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(accuracy_per_class)), accuracy_per_class, tick_label=class_labels)
    plt.title('Accuracy per Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.show()

def main():
    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved model and move it to the device
    model = torch.load('model.pt')
    model = model.to(device)
    model.eval()

    # Get the inference data
    _, test_dataset = getData()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False)

    # Perform inference
    predictions, true_labels = inference(model, test_loader, device)

    # Print inference results
    print("Predictions:", predictions)

    # Calculate accuracy
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    total = len(true_labels)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Plot class distribution (accuracy per class)
    plot_class_distribution(cm, class_labels)

if __name__ == "__main__":
    main()
