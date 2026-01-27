import gzip
import os
from urllib.request import urlretrieve

import numpy as np
from numpy.typing import NDArray

import nanograd
import nanograd.nn as nn
import nanograd.optim as optim


def fetch_mnist() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    os.makedirs("./data", exist_ok=True)
    paths: list[str] = []
    for f in files:
        path = os.path.join("./data", f)
        if not os.path.exists(path):
            print(f"downloading {f}")
            urlretrieve(base_url + f, path)
        paths.append(path)

    def parse(path_img: str, path_lbl: str) -> tuple[NDArray, NDArray]:
        with gzip.open(path_img, "rb") as f_img:
            data = np.frombuffer(f_img.read(), dtype=np.uint8, offset=16)
            data = data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

        with gzip.open(path_lbl, "rb") as f_lbl:
            target = np.frombuffer(f_lbl.read(), dtype=np.uint8, offset=8)

        return data, target

    X_train, Y_train = parse(path_img=paths[0], path_lbl=paths[1])
    X_test, Y_test = parse(path_img=paths[2], path_lbl=paths[3])

    return X_train, Y_train, X_test, Y_test


def main() -> None:
    X_train, Y_train, X_test, Y_test = fetch_mnist()

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(X_train)} examples\n")

    batch_size = 64
    steps = len(X_train) // batch_size

    for epoch in range(3):
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        Y_train = Y_train[perm]

        for i in range(steps):
            start = i * batch_size
            end = start + batch_size

            batch_x = nanograd.tensor(X_train[start:end])
            batch_y = nanograd.tensor(Y_train[start:end])

            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch + 1} | Step {i:3d}/{steps} | Loss {loss.item():.4f}")

    print("\nResult:")
    with nanograd.no_grad():
        test_x = nanograd.tensor(X_test)
        logits = model(test_x)

        predictions = np.argmax(logits.data, axis=1)
        accuracy = (predictions == Y_test).mean()

        print(f"  accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
