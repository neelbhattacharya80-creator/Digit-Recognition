import numpy as np
import random
from sklearn.metrics import classification_report
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# import cupy as np  # GPU based

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Preprocessing
# X_train = np.array(X_train) # use if cupy
# Y_train = np.array(Y_train)


def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((len(y), num_classes))
    one_hot_y[np.arange(len(y)), y] = 1
    return one_hot_y


# Normalize
X_train = (X_train / 255.0).astype(np.float32)
X_test = (X_test / 255.0).astype(np.float32)

# (Batch Size, Channels, Height, Width)
X_train = X_train.reshape(60000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)

Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)


class CNN:

    def __init__(self, k=4, in_channels=1, out_channels=16):  # 16CRP FC2
        # Shape: (num_filters, input_channels, kernel_height, kernel_width)
        self.kernels = np.random.randn(out_channels, in_channels, k, k).astype(
            np.float32
        ) * np.sqrt(2 / (in_channels * k * k))
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self.fc1_W = None  # initializes at the first epoch
        self.fc1_bias = np.zeros((10, 1), dtype=np.float32)

        # velocities
        self.v_kernels = np.zeros_like(self.kernels)
        self.v_bias = np.zeros_like(self.bias)
        self.v_fc1_W = None
        self.v_fc1_bias = np.zeros_like(self.fc1_bias)

    def fit(self, X, Y, lr=0.01, beta=0.9, epochs=32, batch_size=128):
        self.gradient_descent(X, Y, lr, beta, epochs, batch_size)

    def predict(self, X, training=False):
        _, _, _, _, _, probs = self.forward_pass(X, training)
        return np.argmax(probs, axis=1)

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

        # Reshape
        img_array = final_image.reshape(1, 1, 28, 28)
        return o_img, final_image, img_array

    def predict_image(self, path):

        o_img, input_image, img_array = self.preprocess_img(path)

        pred = self.predict(img_array)
        _, _, _, _, _, probs = self.forward_pass(img_array)
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
        plt.imshow(input_image, cmap="gray")
        plt.title(f"Network Input\nPred={pred} ({confidence:.1%})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        # Softmax Distribution
        plt.bar(range(10), probs[0])
        plt.xticks(range(10))
        plt.yticks([i / 10 for i in range(11)])
        plt.xlabel("Digit")
        plt.ylabel("Probability")

        plt.tight_layout()
        plt.show()

    def convolution(self, X, kernels, bias, p=1, s=2):
        # X shape: (Batch, Channels, Height, Width)
        # kernels shape: (Num_Filters, Channels, k, k)
        B, C, H, W = X.shape
        F, _, k, _ = kernels.shape

        output_h = (H - k + (2 * p)) // s + 1
        output_w = (W - k + (2 * p)) // s + 1

        if p > 0:
            X = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

        # Reshape filters into a 2D matrix: (F, C * k * k)
        W_col = kernels.reshape(F, -1)

        Z = np.zeros((B, F, output_h, output_w), dtype=np.float32)

        self.X_cols = []

        for n in range(B):
            # extract valid patches and stack them as columns
            X_col = np.zeros((C * k * k, output_h * output_w), dtype=np.float32)
            col_idx = 0

            for h in range(output_h):
                for w in range(output_w):
                    v_s = h * s
                    v_e = v_s + k
                    h_s = w * s
                    h_e = h_s + k

                    # flatten patch it into a column
                    patch = X[n, :, v_s:v_e, h_s:h_e]
                    X_col[:, col_idx] = patch.reshape(-1)
                    col_idx += 1

            self.X_cols.append(X_col)

            # Shape: (F, C * k * k) @ (C * k * k, output_h * output_w) -> (F, output_h * output_w)
            Z_col = W_col @ X_col + bias.reshape(-1, 1)

            # Reshape the flat result back into the 2D image shape
            Z[n, :, :, :] = Z_col.reshape(F, output_h, output_w)

        return Z  # Shape->(60k,16,14,14)

    def relu(self, Z):
        return np.maximum(0, Z)  # Shape->(60k,16,14,14)

    def relu_derivative(self, z):
        return (z > 0).astype(int)

    def pooling(self, Z, n=2):
        B, K, H, W = Z.shape
        output_h = H // n
        output_w = W // n

        # Reshape the array into 2x2 blocks
        # split Height into (output_h, n) and Width into (output_w, n)
        reshaped = Z.reshape(B, K, output_h, n, output_w, n)

        # Take the maximum along the axes that represent the pool window
        P = np.max(reshaped, axis=(3, 5))
        self.mask = reshaped == P[:, :, :, None, :, None]  # masking
        self.mask = self.mask.reshape(Z.shape)

        return P  # Shape->(60k,16,7,7)

    def flatten(self, P):
        B, C, H, W = P.shape
        if self.fc1_W is None:
            self.fc1_W = np.random.randn(10, C * H * W).astype(np.float32) * np.sqrt(
                2 / (C * H * W)  # Reinitialize based on channels
            )
            self.v_fc1_W = np.zeros_like(self.fc1_W)

        f = P.reshape(B, -1)
        return f  # Shape->(60k,784)

    def fully_connected(self, f):
        B, n = f.shape
        s = self.fc1_W @ f.T + self.fc1_bias  # Shape->(10,60k)
        return s.T  # Shape->(60k,10)

    def softmax(self, s):
        z = s - np.max(s, axis=1, keepdims=True)
        z = np.exp(z)
        h = z / np.sum(z, axis=1, keepdims=True)
        return h  # Shape->(60k,10)

    def forward_pass(self, X, training=True, keep=0.95):
        z = self.convolution(X, self.kernels, self.bias)
        a = self.relu(z)
        p = self.pooling(a)
        f = self.flatten(p)

        # Dropout
        self.keep = keep  # Drop %

        if training:
            # Dropout Mask
            self.dropout_mask = (np.random.rand(*f.shape) < self.keep).astype(
                np.float32
            )
            # Apply mask and scale (Inverted Dropout)
            f = (f * self.dropout_mask) / self.keep

        s = self.fully_connected(f)
        h = self.softmax(s)
        return z, a, p, f, s, h  # Shape->(60k,10)

    def compute_gradients(self, X, Y):  # Backprop
        z, a, p, f, s, h = self.forward_pass(X)
        B, _ = h.shape

        i_error = h - Y  # Shape->(60k,10)

        df = self.fc1_W.T @ i_error.T
        df = df.T  # Shape->(60k,784)

        df = (df * self.dropout_mask) / self.keep  # Dropout backprop

        dW_fc = (i_error.T @ f) / B  # Shape->(10,784)
        db_fc = np.sum(i_error, axis=0, keepdims=True).T / B  # Shape->(10,1)

        dP = df.reshape(p.shape)  # Shape->(60k,16,7,7)
        dP_expanded = np.repeat(
            np.repeat(dP, 2, axis=2), 2, axis=3
        )  # duplicates a value to 2x2

        dA = dP_expanded * self.mask  # unpooling Shape->(60k,16,14,14)
        dZ = dA * self.relu_derivative(z)

        F, C, k, _ = self.kernels.shape
        dK_col = np.zeros((F, C * k * k), dtype=np.float32)

        for n in range(B):  # Bypass Dilated Convolution with im2col
            # Shape->(F, H*W) (no. kernels,no. of patches)
            dZ_col = dZ[n].reshape(F, -1)

            # Error * Input.T
            dK_col += dZ_col @ self.X_cols[n].T

        # Reshape (no. filters,H',W')
        dK = dK_col.reshape(self.kernels.shape) / B
        db = np.sum(dZ, axis=(0, 2, 3)) / B  # bias gradient
        """ debug
        print(z.shape)
        print(a.shape)
        print(p.shape)
        print(f.shape)
        print(s.shape)
        print(h.shape)
        print(df.shape)
        print(dP.shape)
        print(dA.shape)
        print(dZ.shape)
        print(dK.shape)
        """
        return dK, db, dW_fc, db_fc

    def loss(self, X, Y):
        _, _, _, _, _, prob = self.forward_pass(X, training=False)
        loss_ = -(1 / X.shape[0]) * np.sum(Y * np.log(prob + 1e-8))
        return loss_

    def gradient_descent(self, X, Y, lr=0.01, beta=0.9, epochs=32, batch_size=128):
        B, C, H, W = X.shape

        for epoch in range(epochs + 1):
            print(f"current epoch: {epoch}")
            # Shuffle training data
            perm = np.random.permutation(B)
            X_shuffled = X[perm, :, :, :]
            Y_shuffled = Y[perm, :]

            # Mini-batches
            for start in range(0, B, batch_size):

                end = start + batch_size

                X_batch = X_shuffled[start:end, :, :, :]
                Y_batch = Y_shuffled[start:end, :]

                # compute gradients
                dK, db, dW_fc, db_fc = self.compute_gradients(X_batch, Y_batch)

                # Update velocities
                self.v_kernels = (beta * self.v_kernels) + ((1 - beta) * dK)
                self.v_bias = (beta * self.v_bias) + ((1 - beta) * db)
                self.v_fc1_W = (beta * self.v_fc1_W) + ((1 - beta) * dW_fc)
                self.v_fc1_bias = (beta * self.v_fc1_bias) + ((1 - beta) * db_fc)

                # Update (momentum SGD)
                self.kernels -= lr * self.v_kernels
                self.bias -= lr * self.v_bias
                self.fc1_W -= lr * self.v_fc1_W
                self.fc1_bias -= lr * self.v_fc1_bias

            if epoch % 4 == 0:
                preds = self.predict(X_train[:5000], training=False)
                true = np.argmax(Y_train[:5000], axis=1)
                accuracy = np.mean(preds == true) * 100
                print(
                    # f"Epoch {epoch}/{epochs}, "
                    f"Training_Loss: {self.loss(X_batch, Y_batch):.4f} \nTraining_accuracy: {accuracy:.4f}%:"
                )

    def test(self, Y_test, preds):
        print("Predictions :")
        # Y_test = np.argmax(Y_test, axis=1).get()  # Convert to cpu to support libraries
        # preds = preds.get()   # use if cupy
        true = np.argmax(Y_test, axis=1)
        for pred, actual in random.sample(list(zip(preds, true)), 10):
            print(f"Predictions: {pred} True Value: {actual}")
        accuracy = np.mean(preds == true) * 100
        print("\nModel Evaluation-")
        print(f"accuracy: {accuracy}")
        print(classification_report(true, preds))

    def save_model(self, filename="cnn_model.npz"):
        np.savez(
            filename,
            kernels=self.kernels,
            bias=self.bias,
            fc1_W=self.fc1_W,
            fc1_bias=self.fc1_bias,
        )
        print(f"Model successfully saved to {filename}")

    def load_model(self, filename="cnn_model.npz"):
        data = np.load(filename)
        self.kernels = data["kernels"]
        self.bias = data["bias"]
        self.fc1_W = data["fc1_W"]
        self.fc1_bias = data["fc1_bias"]

        print(f"Model successfully loaded from {filename}")


cnn = CNN()

path = r"C:\Users\neelb\Documents\CS\Projects\Digit_Recognition\cnn_model.npz"
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

# cnn.fit(X_train, Y_train)
# cnn.save_model(path)
cnn.load_model(path)

preds = cnn.predict(X_test)
cnn.test(Y_test, preds)

cnn.predict_image(img_paths[6])
