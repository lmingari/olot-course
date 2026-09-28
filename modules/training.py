from abc import ABC, abstractmethod
import torch
import pandas as pd

class Trainer(ABC):
    """Base class for training PyTorch models.

    Subclasses must implement ``get_loss()``. The trainer handles the
    training and validation loops.

    Example
    -------
    >>> class MyTrainer(Trainer):
    ...     def get_loss(self, batch):
    ...         x, y = batch
    ...         prediction = self.model(x)
    ...         return self.criterion(prediction, y)
    ...
    >>> trainer = MyTrainer(model, criterion, optimizer)
    >>> history = trainer.fit(train_loader, val_loader, epochs=20)
    """
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    @abstractmethod
    def get_loss(self, batch):
        """Compute the loss for a batch."""
        ...

    def train_epoch(self, dataloader):
        self.model.train()

        total_loss = 0.0

        for batch in dataloader:
            self.optimizer.zero_grad()

            loss = self.get_loss(batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate_epoch(self, dataloader):
        self.model.eval()

        total_loss = 0.0

        for batch in dataloader:
            loss = self.get_loss(batch)
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader, epochs, log_every=100):
        history = []
    
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate_epoch(val_loader)
    
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
    
            if (epoch + 1) % log_every == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:4d} | "
                    f"train loss: {train_loss:.4f} | "
                    f"val loss: {val_loss:.4f}"
                )
        print("Done!")
    
        return pd.DataFrame(history)