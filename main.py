import torch
from torch.utils.data import DataLoader
from dataset import EasyQADataset
from models import EasyQAEarlyFusionNetwork, EasyQAMidFusionNetwork
from utils import accuracy_score_func
import pandas as pd
from config import *

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for batch in dataloader:
        images, texts, labels = batch['image'], batch['text'], batch['label']
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        accuracy = accuracy_score_func(outputs, labels)
        total_accuracy += accuracy
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in dataloader:
            images, texts, labels = batch['image'], batch['text'], batch['label']
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            accuracy = accuracy_score_func(outputs, labels)
            total_accuracy += accuracy
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy

def main():
    device = torch.device(DEVICE)
    # Load data
    train_df = pd.read_csv(f'{DATA_DIR}/{TRAIN_DATA_FILE}')
    val_df = pd.read_csv(f'{DATA_DIR}/{VALIDATION_DATA_FILE}')

    # Initialize datasets
    train_dataset = EasyQADataset(train_df, tokenizer, image_transform, device)
    val_dataset = EasyQADataset(val_df, tokenizer, image_transform, device)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = EasyQAEarlyFusionNetwork().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

    print("Training completed!")

if __name__ == '__main__':
    main()

