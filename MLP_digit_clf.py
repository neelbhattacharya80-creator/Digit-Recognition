import numpy as np
import random
from sklearn.metrics import classification_report
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Preprocessing

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0


def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((num_classes, len(y)))
    one_hot_y[y, np.arange(len(y))] = 1
    return one_hot_y


X_train, X_test = X_train.T, X_test.T

Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)


class Neural_Network:
    def __init__(self, lr=0.01, lambda_=0.001, layers=[784, 256, 128, 10]):
        self.lr = lr
        self.input_size = layers[0]
        self.layers = layers
        self.lambda_ = lambda_
        self.output_size = layers[-1]
        self.weights = [
            np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2 / layers[i])
            for i in range(len(layers) - 1)
        ]
        self.bias = [np.zeros((layers[i + 1], 1)) for i in range(len(layers) - 1)]

    def fit(self, X, Y, epochs=20, batch_size=64):
        self.gradient_descent(X, Y, epochs, batch_size)

    def predict(self, X):
        p = self.forward_pass(X)
        preds = np.argmax(p, axis=0)
        return preds

    def preprocess_img(self, path):
        o_img = Image.open(path).convert("L")
        img_array = np.array(o_img)

        # Invert to a 0.0 - 1.0 scale
        img_inverted = 1.0 - (img_array / 255.0)

        # Dynamic Thresholding
        # Calculate the median pixel value, which represents the paper background
        bg_value = np.median(img_inverted)

        # Wipe out the background and any shadows/noise slightly darker than it
        img_inverted[img_inverted < bg_value + 0.10] = 0.0

        # Multiply the surviving pixels by a high factor to make faint grey lines solid white,
        # then clip at 1.0 so we don't break our 0.0-1.0 scale constraint.
        img_inverted = np.clip(img_inverted * 5.0, 0.0, 1.0)

        # Find the tight bounding box of the actual digit using NumPy
        # rows and columns that have at least one pixel > 0
        rows = np.any(img_inverted > 0, axis=1)
        cols = np.any(img_inverted > 0, axis=0)

        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Crop
            cropped = img_inverted[rmin : rmax + 1, cmin : cmax + 1]
        else:
            cropped = img_inverted

        # Convert back to PIL

        cropped_image = Image.fromarray((cropped * 255).astype(np.uint8))
        cropped_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

        resized_array = np.array(cropped_image) / 255.0

        # Create the final 28x28 pure black canvas
        final_image = np.zeros((28, 28))

        # Paste the resized digit perfectly in the center
        h, w = resized_array.shape
        y_off = (28 - h) // 2
        x_off = (28 - w) // 2
        final_image[y_off : y_off + h, x_off : x_off + w] = resized_array

        # Boost the signal so the brightest pixel is exactly 1.0
        if np.max(final_image) > 0:
            final_image = final_image / np.max(final_image)

        # Flatten
        img_flattened = final_image.reshape(784, 1)
        return o_img, final_image, img_flattened

    def predict_image(self, path):

        o_img, input_img, img_flattened = self.preprocess_img(path)

        # Pass the flattened image to predict and forward_pass
        pred = self.predict(img_flattened)[0]
        probs = self.forward_pass(img_flattened)
        confidence = np.max(probs)

        # Plot
        plt.figure(figsize=(12, 4.5))

        plt.subplot(1, 3, 1)
        # Original Image
        plt.imshow(o_img, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        # Network Input
        plt.imshow(input_img, cmap="gray")
        plt.title(f"Network Input\nPred={pred} ({confidence:.1%})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        # Softmax Distribution
        plt.bar(range(10), probs[:, 0])
        plt.xticks(range(10))
        plt.yticks([i / 10 for i in range(11)])
        plt.xlabel("Digit")
        plt.ylabel("Probability")

        plt.tight_layout()
        plt.show()

    def forward_pass(self, X):
        self.z = []
        self.a = [X]
        for l in range(len(self.weights)):
            self.z.append(self.weights[l] @ self.a[l] + self.bias[l])  # W1X + b1

            if l == (len(self.weights) - 1):
                self.a.append(self.softmax(self.z[l]))  #  W3a2+b3
            else:
                self.a.append(self.relu(self.z[l]))  # W2a1+b2

        return self.a[-1]

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(int)

    def softmax(self, z):  # Compute Probabilities based of aL
        z = z - np.max(z, axis=0, keepdims=True)
        z = np.exp(z)
        return z / np.sum(z, axis=0, keepdims=True)

    def loss(self, X, Y):
        prob = self.forward_pass(X)
        self.loss_ = -(1 / Y.shape[1]) * np.sum(Y * np.log(prob + 1e-8))

        # L2 Regularization
        self.loss_ += (self.lambda_ / (2 * Y.shape[1])) * sum(
            np.sum(W**2) for W in self.weights
        )
        return self.loss_

    def error_(self, X, Y):
        error = []
        prob = self.forward_pass(X)
        i_error = prob - Y
        for l in range(len(self.weights) - 1):
            if l == 0:
                error.append(i_error)
            e = (
                self.weights[-(l + 1)].T
                @ error[-1]
                * self.relu_derivative(self.z[-(l + 2)])
            )
            error.append(e)
        return error

    def compute_gradients(self, X, Y):
        error = self.error_(X, Y)
        self.dW = []
        self.db = []
        m = X.shape[1]
        for l in range(len(self.weights)):
            dW = (1 / m) * error[l] @ self.a[-(l + 2)].T
            db = (1 / m) * np.sum(error[l], axis=1, keepdims=True)
            dW += (self.lambda_ / m) * self.weights[-(l + 1)]
            self.dW.append(dW)
            self.db.append(db)
        self.dW.reverse()
        self.db.reverse()
        return self.dW, self.db

    def gradient_descent(self, X, Y, epochs, batch_size=64):

        m = X.shape[1]

        for epoch in range(epochs + 1):

            # Shuffle training data
            perm = np.random.permutation(m)
            X_shuffled = X[:, perm]
            Y_shuffled = Y[:, perm]

            # Mini-batches
            for start in range(0, m, batch_size):

                end = start + batch_size

                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]

                dW, db = self.compute_gradients(X_batch, Y_batch)

                for l in range(len(self.weights)):
                    self.weights[l] -= self.lr * dW[l]
                    self.bias[l] -= self.lr * db[l]
            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch}/{epochs}, " f"Training_Loss: {self.loss(X, Y):.4f}"
                )

    def test(self, Y_test, preds):
        print("Predictions :")
        Y_test = np.argmax(Y_test, axis=0)
        for pred, actual in random.sample(list(zip(preds, Y_test)), 10):
            print(f"Predictions: {pred} True Value: {actual}")

        print("\nModel Evaluation-")
        print(classification_report(Y_test, preds))

    def save_model(self, filename="model.npz"):
        np.savez(filename, *self.weights, *self.bias)

    def load_model(self, filename="model.npz"):
        data = np.load(filename)

        n_weights = len(self.weights)

        self.weights = [data[f"arr_{i}"] for i in range(n_weights)]

        self.bias = [data[f"arr_{i+n_weights}"] for i in range(len(self.bias))]


model = Neural_Network()

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


# model.fit(X_train, Y_train, epochs=50)

# model.save_model(r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\mnist_model.npz")
model.load_model(
    r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\mnist_model.npz"
)

preds = model.predict(X_test)


model.test(Y_test, preds)

model.predict_image(img_paths[0])
