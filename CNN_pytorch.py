import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import random
from sklearn.metrics import classification_report
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize and reshape
X_train = (X_train / 255.0).astype(np.float32).reshape(-1, 1, 28, 28)
X_test = (X_test / 255.0).astype(np.float32).reshape(-1, 1, 28, 28)


class CNN(nn.Module):
    def __init__(self):
        # Initialize the nn.Module parent class first
        super().__init__()

        # Architecture
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.05),
            # FC1: Input [Batch, 288] -> Output [Batch, 128]
            nn.Linear(in_features=288, out_features=128),
            nn.ReLU(),
            # FC2: Input [Batch, 128] -> Output [Batch, 64]
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            # FC3: Input [Batch, 64] -> Output [Batch, 10]
            nn.Linear(in_features=64, out_features=10),
        )

        # Setup Hardware Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move all registered layers to the GPU/CPU

        # Optimizer & Loss
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    # forward pass
    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits

    # Training
    def fit(self, X, Y, epochs=32, batch_size=128):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs + 1):
            self.train()  # Set self to training mode
            epoch_loss = 0.0

            for batch_X, batch_Y in loader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(batch_X)  # Calls self.forward(batch_X) natively
                loss = self.criterion(outputs, batch_Y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if epoch % 4 == 0:
                acc = self.evaluate_accuracy(X[:5000], Y[:5000])
                avg_loss = epoch_loss / len(loader)
                print(
                    f"Epoch {epoch} | Training Loss: {avg_loss:.4f} | Training Accuracy: {acc:.2f}%"
                )

    def evaluate_accuracy(self, X, Y):
        preds = self.predict(X)
        return np.mean(preds == Y) * 100

    def predict(self, X):
        self.eval()  # Set self to evaluation mode
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self(X_tensor)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def get_probabilities(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self(X_tensor)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def test(self, Y_test, preds):
        print("Predictions :")
        for pred, actual in random.sample(list(zip(preds, Y_test)), 10):
            print(f"Prediction: {pred} True Value: {actual}")

        accuracy = np.mean(preds == Y_test) * 100
        print("\nModel Evaluation-")
        print(f"Accuracy: {accuracy:.2f}%")
        print(classification_report(Y_test, preds))

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print(f"Model successfully saved to {filename}")

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))
        self.eval()
        print(f"Model successfully loaded from {filename}")

    # Image Preprocessing
    def preprocess_img(self, path):
        o_img = Image.open(path).convert("L")
        img_array = np.array(o_img)

        img_inverted = 1.0 - (img_array / 255.0)
        bg_value = np.median(img_inverted)
        img_inverted[img_inverted < bg_value + 0.10] = 0.0
        img_inverted = np.clip(img_inverted * 5.0, 0.0, 1.0)

        rows = np.any(img_inverted > 0, axis=1)
        cols = np.any(img_inverted > 0, axis=0)

        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            cropped = img_inverted[rmin : rmax + 1, cmin : cmax + 1]
        else:
            cropped = img_inverted

        cropped_image = Image.fromarray((cropped * 255).astype(np.uint8))
        cropped_image.thumbnail((20, 20), Image.Resampling.LANCZOS)
        resized_array = np.array(cropped_image) / 255.0

        final_image = np.zeros((28, 28))
        h, w = resized_array.shape
        y_off = (28 - h) // 2
        x_off = (28 - w) // 2
        final_image[y_off : y_off + h, x_off : x_off + w] = resized_array

        if np.max(final_image) > 0:
            final_image = final_image / np.max(final_image)

        img_array = final_image.reshape(1, 1, 28, 28)
        return o_img, final_image, img_array

    def predict_image(self, path):
        o_img, input_image, img_array = self.preprocess_img(path)

        pred = self.predict(img_array)[0]
        probs = self.get_probabilities(img_array)[0]
        confidence = np.max(probs)

        plt.figure(figsize=(12, 4.5))

        plt.subplot(1, 3, 1)
        plt.imshow(o_img, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(input_image, cmap="gray")
        plt.title(f"Network Input\nPred={pred} ({confidence:.1%})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.bar(range(10), probs)
        plt.xticks(range(10))
        plt.yticks([i / 10 for i in range(11)])
        plt.xlabel("Digit")
        plt.ylabel("Probability")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cnn = CNN()

    path = (
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\cnn_model_pytorch.pth"
    )
    img_paths = [
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\0.jpg",
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\1.jpg",
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\2.jpg",
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\3.jpg",
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\4.jpg",
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\5.jpg",
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\6.jpg",
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\7.jpg",
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\8.jpg",
        r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\sample_images\9.jpg",
    ]

    # Train
    # cnn.fit(X_train, Y_train)
    # cnn.save_model(path)

    # Load and evaluate
    cnn.load_model(path)
    preds = cnn.predict(X_test)
    cnn.test(Y_test, preds)
    cnn.predict_image(img_paths[7])
