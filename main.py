import tkinter as tk
from pathlib import Path
from tkinter import filedialog
import kaggle
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import ImageOps

IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 10
LABEL_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Using device: {device}")

kaggle.api.authenticate()
path = kagglehub.dataset_download("zalando-research/fashionmnist")
root = Path(kagglehub.dataset_download("zalando-research/fashionmnist"))

print(f"Dataset is in: {root}")
train_csv = next(root.rglob("*train*.csv"))
test_csv  = next(root.rglob("*test*.csv"))
train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)


train_images = train_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.float32) / 255.0
test_images = test_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.float32) / 255.0

trainset = torch.utils.data.TensorDataset(
    torch.tensor(train_images, dtype=torch.float32), 
    torch.tensor(train_df.iloc[:, 0].values, dtype=torch.long)
)
testset = torch.utils.data.TensorDataset(
    torch.tensor(test_images, dtype=torch.float32), 
    torch.tensor(test_df.iloc[:, 0].values, dtype=torch.long)
)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=False)
testloader  = DataLoader(testset, batch_size=128, shuffle=False)

dataiter = iter(trainloader)
images, labels = next(dataiter)
print(images.shape)
print(labels.shape)

import torch.nn as nn
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding =1),
            nn.SiLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.SiLU(),
            nn.Linear(128,10)
        )

    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.network(x)

model = CNNModel().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
images, labels = next(iter(trainloader))
images, labels = images.to(device), labels.to(device)
logits = model(images)
loss = criterion(logits, labels)

import torch.optim as optim
from time import time

optimizer = optim.Adam(model.parameters(), lr = 0.001) #If you use SGD, Remember to tweak the momentum value to 0.9 in the **Kwargs
epochs = int(input("Enter the number of epochs (Default: 50) "))
start_time = time()

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {:.4f}".format(
            e, running_loss / len(trainloader)
        ))

print("\nTraining Time in minutes: {:.2f}".format((time() - start_time)/60))

def view_classify(img, ps, version="Fashion"):
    ps = ps.cpu().data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 8), ncols=2)
    
    # Image display
    ax1.imshow(img.cpu().numpy().squeeze(), cmap='gray')
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14)
    
    # Probability bar chart
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.05)
    ax2.set_yticks(np.arange(10))
    
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10), fontsize=12)
    elif version == "Fashion":
        ax2.set_yticklabels(LABEL_NAMES, fontsize=10)
    
    ax2.set_title('Class Probability', fontsize=14)
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel('Probability', fontsize=12)
    

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
    plt.show()


images, labels = next(iter(testloader))
images, labels = images.to(device), labels.to(device)
img = images[0].unsqueeze(0)
model.eval()
with torch.no_grad():
    logits = model(img)
    probs = torch.softmax(logits, dim=1)
probab = probs.cpu().numpy()[0]
print("Predicted class:", probab.argmax())
view_classify(img.squeeze(), probs)


def select_and_classify_image():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
    )

    if file_path:
        image = Image.open(file_path).convert('L')
        image = ImageOps.invert(image)  # Invert colors (white background to black)
        image = image.resize((28, 28))
        
        # Display the input image
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.title("Input Image")
        plt.axis('off')
        plt.show()
        
        # Transform and classify
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        print(f"Predicted: {LABEL_NAMES[predicted.item()]} with confidence: {confidence.item() * 100:.2f}%")
        
        # Show classification results
        view_classify(image_tensor.squeeze(), probabilities)
    
    root.destroy()




print("\nSelect an image to classify...")
while True:
 select_and_classify_image()
 if input("Do you want to try again? (y/n): ").lower() != 'y':
     break