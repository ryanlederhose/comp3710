import torch
import torch.nn as nn
from torch import optim
from torch.hub import tqdm
from dataset import DataLoader
from modules import ViT
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

class HyperParameters(object):
    '''
    HyperParameters
    
    This class defines all the hyperparameters for 
    the vision transformer
    '''

    def __init__(self) -> None:
        self.patch_size = 8
        self.mlp_dim = 1000
        self.head_dim = 300
        self.n_channels = 3
        self.num_encoders = 4
        self.num_heads = 4
        self.dropout = 0.2
        self.num_classes = 2
        self.epochs = 60
        self.lr = 3e-4
        self.weight_decay = 0
        self.batch_size = 32
        self.dry_run = False
        self.hidden_size = 64

def check_accuracy(loader, model, device):
    '''
    Check the accuracy of the model on the given dataloader
    '''
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

    model.train()
    accuracy = num_correct / num_samples
    return accuracy

def train_model(args, train_loader, epoch, device, model, criterion, optimizer):
    '''
    Train a specified model
    '''

    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    tk = tqdm(train_loader, desc="EPOCH" + "[TRAIN]" 
                + str(epoch + 1) + "/" + str(args.epochs))
    
    for batch_idx, (data, targets) in enumerate(tk):
        # Get data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward propogation
        scores = model(data)
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)
        loss = criterion(scores, targets)

        # Back propogation
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        tk.set_postfix({"Loss": "%6f" % float(total_loss / (batch_idx + 1))})
    
    accuracy = num_correct / num_samples
    return ((total_loss / len(train_loader)), accuracy)

def validate_model(args, val_loader, model, epoch, device, criterion):
    '''
    Evaluate a specified model
    '''

    model.eval()
    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    tk = tqdm(val_loader, desc="EPOCH" + "[VALID]" + str(epoch + 1) + "/" + str(args.epochs))

    with torch.no_grad():
        for t, (data, targets) in enumerate(tk):
            data, targets = data.to(device), targets.to(device)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
            loss = criterion(scores, targets)

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})

    accuracy = num_correct / num_samples
    print(f"Accuracy on validation set: {accuracy *100:.2f}")
    model.train()
    return ((total_loss / len(val_loader)), accuracy)

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Training and Validation Set Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(["Training", "Validation"])
    plt.show()

def plot_accuracy(train_accuracy, val_accuracy):
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('Training and Validation Set Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(["Training", "Validation"])
    plt.show()


def main():

    args = HyperParameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dl = DataLoader(batch_size=args.batch_size)
    train_loader = dl.get_training_loader()
    test_loader = dl.get_test_loader()
    valid_loader = dl.get_valid_loader()
    trainLoss = 0.0
    trainAccuracy = 0.0
    trainLossList = []
    trainAccuracyList = []
    validLoss = 0.0
    validAccuracy = 0.0
    validLossList = []
    validAccuracyList = []

    model = ViT(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, min_lr=1e-8, 
                                               verbose=True)
    model.train()

    for epoch in range(args.epochs):
        trainLoss, trainAccuracy = train_model(args, train_loader, epoch, device, model, criterion, optimizer)
        validLoss, validAccuracy = validate_model(args, valid_loader, model, epoch, device, criterion)
        scheduler.step(trainLoss)

        trainLossList.append(trainLoss)
        validLossList.append(validLoss)
        trainAccuracyList.append(trainAccuracy)
        validAccuracyList.append(validAccuracy)
    
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")
    
    plot_losses(trainLossList, validLossList)
    plot_accuracy(trainAccuracyList, validAccuracyList)

if __name__ == "__main__":
    main()