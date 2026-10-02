from abc import ABC, abstractmethod
import torch
import pandas as pd

class Trainer(ABC):
    """Base class for training PyTorch models.

    Subclasses must implement ``get_batch_results()``. The trainer handles the
    training and validation loops.

    Example
    -------
    >>> class MyTrainer(Trainer):
    ...     def get_batch_results(self, batch):
    ...         x, y = batch
    ...         prediction = self.model(x)
    ...         loss = self.criterion(prediction, y)
    ...         return {'loss': loss}
    ...
    >>> trainer = MyTrainer(model, criterion, optimizer)
    >>> history = trainer.fit(train_loader, val_loader, epochs=20)
    """
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    @abstractmethod
    def get_batch_results(self, batch):
        """Return loss and metrics for a batch."""
        ...

    def train_epoch(self, dataloader):
        self.model.train()

        totals = {}

        for batch in dataloader:
            self.optimizer.zero_grad()

            results = self.get_batch_results(batch)

            results["loss"].backward()
            self.optimizer.step()

            for name, value in results.items():
                value = value.item()
                totals[name] = totals.get(name, 0) + value

        return {
            name: value / len(dataloader)
            for name, value in totals.items()
        }
        
    @torch.no_grad()
    def evaluate_epoch(self, dataloader):
        self.model.eval()

        totals = {}

        for batch in dataloader:
            results = self.get_batch_results(batch)

            for name, value in results.items():
                value = value.item()
                totals[name] = totals.get(name, 0) + value

        return {
            name: value / len(dataloader)
            for name, value in totals.items()
        }

    def _print_log(self, results):
        print(
            f"Epoch {results['epoch']:4d} | "
            f"train loss: {results['train_loss']:.4f} | "
            f"val loss: {results['val_loss']:.4f}"
        )

    def fit(self, train_loader, val_loader, epochs):
        """ Implement a generic training loop """
        history = []

        log_every = max(1, epochs // 10)
    
        for epoch in range(epochs):
            train = self.train_epoch(train_loader)
            val   = self.evaluate_epoch(val_loader)
    
            results = {
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in train.items()},
                **{f"val_{k}": v for k, v in val.items()},
            }
            history.append(results)
    
            if (epoch + 1) % log_every == 0 or epoch == 0:
                self._print_log(results)
                
        print("Done!")
    
        return pd.DataFrame(history)

class SupervisedTrainer(Trainer):
    def get_batch_results(self, batch):
        x, y = batch
        prediction = self.model(x)
        loss = self.criterion(prediction, y)
        return {'loss': loss}        