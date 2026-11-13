"""AI algorithm: MLP, supervised training, and inference across architectures.

The dataset is synthetic and local. Classes do not represent MNIST digits.
Only this module handles loss, gradients, and weight updates.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def synthetic_dataset(features=784, classes=10, seed=42):
    """Create independent training, validation, and test splits for a controlled problem.

    Each class has a center in the first 32 features. The remaining features
    are noise. Class centers are shared across all three splits.
    No test sample is used for training.
    """
    if features < 1 or classes < 2:
        raise ValueError("Provide at least one feature and two classes.")

    rng = np.random.default_rng(seed)
    signal_features = min(32, features)
    centers = rng.normal(0, 2, (classes, signal_features)).astype(np.float32)
    subsets = []

    for size in (3000, 600, 600):
        labels = rng.permutation(np.arange(size) % classes)
        inputs = rng.normal(0, 1, (size, features)).astype(np.float32)
        inputs[:, :signal_features] += centers[labels]
        subsets.append((inputs, labels))

    return subsets


def softmax(logits):
    """Subtracting the largest logit keeps exponentiation numerically stable."""
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


class MLP:
    """Dense layers with ReLU hidden activations and softmax output."""

    def __init__(self, dimensions=(784, 128, 64, 10), seed=42):
        valid = all(isinstance(n, int) and n >= 1 for n in dimensions)
        if len(dimensions) < 2 or not valid:
            raise ValueError("Dimensions must be positive integers.")

        self.dimensions = tuple(dimensions)
        rng = np.random.default_rng(seed)

        # He initialization maintains activation scale with ReLU.
        self.weights = []
        for inputs, outputs in zip(dimensions[:-1], dimensions[1:]):
            scale = np.sqrt(2 / inputs)
            weights = rng.normal(size=(inputs, outputs)) * scale
            self.weights.append(weights.astype(np.float32))
        self.biases = [np.zeros(n, dtype=np.float32) for n in dimensions[1:]]

    def forward(self, inputs):
        """NumPy reference for training and validating native implementations."""
        activation = np.asarray(inputs)
        if activation.ndim != 2 or activation.shape[1] != self.dimensions[0]:
            raise ValueError("Input does not match the network dimensions.")

        # Intermediate activations are also used during backpropagation.
        activations = [activation]

        for layer, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            logits = activation @ weights
            logits = logits + bias

            # Hidden layers use ReLU; the final layer produces probabilities.
            if layer == len(self.weights) - 1:
                activation = softmax(logits)
            else:
                activation = np.maximum(0, logits)
            activations.append(activation)

        return activations

    def predict_proba(self, inputs):
        return self.forward(inputs)[-1]

    def save(self, path):
        content = {"dimensions": np.array(self.dimensions)}
        content.update({f"W{i}": value for i, value in enumerate(self.weights)})
        content.update({f"b{i}": value for i, value in enumerate(self.biases)})
        np.savez(path, **content)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as content:
            model = cls(tuple(int(n) for n in content["dimensions"]))
            model.weights = [content[f"W{i}"].copy() for i in range(len(model.weights))]
            model.biases = [content[f"b{i}"].copy() for i in range(len(model.biases))]
        return model


def train_model(dimensions=(784, 128, 64, 10), epochs=12, output=ROOT / "results"):
    """Train with Adam and select the checkpoint using validation only."""
    if epochs < 1:
        raise ValueError("The number of epochs must be positive.")

    model = MLP(dimensions)
    train, validation_set, test = synthetic_dataset(dimensions[0], dimensions[-1])
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = train, validation_set, test
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(43)
    parameters = model.weights + model.biases
    moments = [np.zeros_like(value) for value in parameters]
    variances = [np.zeros_like(value) for value in parameters]
    history = []
    best_accuracy = -1
    step = 0

    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        total_loss = 0.0

        for offset in range(0, len(order), 128):
            indices = order[offset : offset + 128]
            activations = model.forward(x_train[indices])
            probabilities = activations[-1]
            labels = y_train[indices]
            # Probability assigned to the correct class; clipping avoids log(0).
            correct = probabilities[np.arange(len(indices)), labels]
            total_loss -= np.log(np.maximum(correct, 1e-12)).sum()

            # Softmax with cross entropy: output gradient = p - target.
            gradient = probabilities.copy()
            gradient[np.arange(len(indices)), labels] -= 1
            gradient /= len(indices)
            weight_gradients = [None] * len(model.weights)
            bias_gradients = [None] * len(model.biases)

            for layer in reversed(range(len(model.weights))):
                weight_gradients[layer] = activations[layer].T @ gradient
                bias_gradients[layer] = gradient.sum(axis=0)
                if layer > 0:
                    # The ReLU derivative is 1 for positive activations and 0 elsewhere.
                    relu_derivative = activations[layer] > 0
                    gradient = (gradient @ model.weights[layer].T) * relu_derivative

            # Adam with bias correction and a learning rate of 0.001.
            step += 1
            gradients = weight_gradients + bias_gradients
            for index, (parameter, grad) in enumerate(zip(parameters, gradients)):
                moments[index] = 0.9 * moments[index] + 0.1 * grad
                variances[index] = 0.999 * variances[index] + 0.001 * grad**2
                parameter -= 0.001 * (moments[index] / (1 - 0.9**step)) / (np.sqrt(variances[index] / (1 - 0.999**step)) + 1e-8)

        predicted = model.predict_proba(x_val).argmax(axis=1)
        validation = float((predicted == y_val).mean())
        loss = float(total_loss / len(x_train))
        history.append({"epoch": epoch + 1, "loss": loss, "validation_accuracy": validation})
        print(f"Epoch {epoch + 1}: loss={loss:.4f}; validation={validation:.2%}", flush=True)

        if validation > best_accuracy:
            best_accuracy = validation
            model.save(output / "ai_model.npz")

    # The test split is used only after selecting the best checkpoint.
    model = MLP.load(output / "ai_model.npz")
    predictions = model.predict_proba(x_test).argmax(axis=1)
    metrics = {"dataset": "synthetic: class centers + Gaussian noise; not MNIST", "dimensions": list(dimensions), "train_samples": len(x_train), "validation_samples": len(x_val), "test_samples": len(x_test), "test_accuracy": float((predictions == y_test).mean()), "seed": 42, "history": history}
    report = json.dumps(metrics, ensure_ascii=False, indent=2)
    (output / "ai_metrics.json").write_text(report, encoding="utf-8")
    np.savez(output / "ai_test.npz", inputs=x_test, labels=y_test)
    print(f"Synthetic test accuracy: {metrics['test_accuracy']:.2%}")
    return model, metrics


def export_native(model_folder=ROOT / "results", output=ROOT / "data" / "native"):
    """Export identical FP32 values to C++ and CUDA with a simple binary header.

    Each file contains two little-endian uint32 values (rows, columns), followed
    by row-major little-endian float32 values. No C++ libraries are required.
    """
    import struct

    model_folder = Path(model_folder)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = MLP.load(model_folder / "ai_model.npz")

    def write_matrix(name, values):
        values = np.asarray(values, dtype="<f4")
        if values.ndim == 1:
            values = values[None, :]
        with (output / name).open("wb") as stream:
            stream.write(struct.pack("<II", *values.shape))
            stream.write(values.tobytes(order="C"))

    for index, (weights, bias) in enumerate(zip(model.weights, model.biases)):
        write_matrix(f"W{index}.bin", weights)
        write_matrix(f"b{index}.bin", bias)

    with np.load(model_folder / "ai_test.npz", allow_pickle=False) as test:
        inputs = test["inputs"].astype(np.float32)
        labels = test["labels"]

    write_matrix("inputs.bin", inputs)
    write_matrix("labels.bin", labels[:, None])
    write_matrix("probabilities.bin", model.predict_proba(inputs))
    (output / "model.txt").write_text(str(len(model.weights)) + "\n", encoding="ascii")
    print(f"FP32 model exported to {output}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--native-output", type=Path, default=ROOT / "data" / "native")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if not args.export_only:
        train_model(tuple(settings["dimensions"]), args.epochs, args.output)
    export_native(args.output, args.native_output)
