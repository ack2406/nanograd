import nanograd
import nanograd.nn as nn
import nanograd.optim as optim


def main():
    X = nanograd.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=nanograd.float32)
    y = nanograd.tensor([[0], [1], [1], [0]], dtype=nanograd.float32)

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    )

    optimizer = optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()

    print(f"Training on {len(X.data)} examples\n")

    for epoch in range(10_001):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch [{epoch:5d}/10000] | Loss [{loss.item():.6f}]")

    with nanograd.no_grad():
        pred = model(X)

        print("\nRaw results:")
        print(pred)

        print("\nRounded results:")
        print(pred.round())


if __name__ == "__main__":
    main()
