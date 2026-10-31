
import torch
import torch.optim as optim
from siamese_model import SiameseNetwork
from utils import preprocess
import random
from pathlib import Path
import os

def train_model():
    anchor_dir = Path("data/anchor")
    positive_dir = Path("data/positive")
    negative_dir = Path("data/negative")

    anchors = list(anchor_dir.glob("*"))
    positives = list(positive_dir.glob("*"))
    negatives = list(negative_dir.glob("*"))

    model = SiameseNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CosineEmbeddingLoss()

    epochs = 3
    for epoch in range(epochs):
        running_loss = 0.0
        for _ in range(10):  # small dummy training
            anchor = preprocess(random.choice(anchors))
            if random.random() > 0.5:
                positive = preprocess(random.choice(positives))
                label = torch.tensor([1.0])
            else:
                positive = preprocess(random.choice(negatives))
                label = torch.tensor([-1.0])

            optimizer.zero_grad()
            output1, output2 = model(anchor, positive)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/siamese_model.pth")
    return running_loss / 10
