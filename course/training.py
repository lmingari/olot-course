from abc import ABC, abstractmethod
import torch
import pandas as pd


class Trainer(ABC):
    """Base class for training PyTorch models (CPU or GPU).

    Subclasses must implement ``get_batch_results()``. The trainer handles the
    training and validation loops and moves the model and every batch to
    ``device``, so ``get_batch_results()`` always receives tensors that already
    live on the right device.

    Example
    -------
    >>> class MyTrainer(Trainer):
    ...     def get_batch_results(self, batch):
    ...         x, y = batch
    ...         prediction = self.model(x)
    ...         loss = self.criterion(prediction, y)
    ...         return {'loss': loss}
    ...
    >>> trainer = MyTrainer(model, criterion, optimizer)   # uses the GPU if available
    >>> history = trainer.fit(train_loader, val_loader, epochs=20)
    """
    def __init__(self, model, criterion, optimizer, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

    def _to_device(self, batch):
        """Move a tensor, or a (nested) list/tuple of tensors, to self.device."""
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            moved = [self._to_device(b) for b in batch]
            return moved if isinstance(batch, list) else tuple(moved)
        return batch

    @abstractmethod
    def get_batch_results(self, batch):
        """Return loss and metrics for a batch (a dict of scalar tensors)."""
        ...

    def train_epoch(self, dataloader):
        self.model.train()

        totals = {}

        for batch in dataloader:
            batch = self._to_device(batch)
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
            batch = self._to_device(batch)
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


class FlowMatchingTrainer(Trainer):
    """Trainer for (conditional) flow matching with a linear interpolation path.

    Each batch of data ``x1`` is turned into a training example:

        x0     ~ N(0, I)                     noise,                 (B, C, H, W)
        t      ~ U(0, 1)                     one time per sample,   (B,)
        x_t    = (1 - t) * x0 + t * x1       point on the path,     (B, C, H, W)
        target = x1 - x0                     velocity of the path,  (B, C, H, W)

    and the model learns to predict the target: ``criterion(model(x_t, t), target)``.

    The dataloader must yield batches of images: a tensor ``x1`` of shape
    (B, C, H, W).

    Example
    -------
    >>> model = FlowUNet(channels=1)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> trainer = FlowMatchingTrainer(model, nn.MSELoss(), optimizer)
    >>> history = trainer.fit(train_loader, val_loader, epochs=50)
    """
    def get_batch_results(self, batch):
        """batch: tensor of images (B, C, H, W), already on the device."""
        x1 = batch                                                    # (B, C, H, W)

        x0 = torch.randn_like(x1)                                     # (B, C, H, W)
        t = torch.rand(x1.shape[0], device=x1.device)                 # (B,)
        t_ = t[:, None, None, None]                                   # (B, 1, 1, 1) to broadcast

        xt = (1 - t_) * x0 + t_ * x1                                  # (B, C, H, W)
        target = x1 - x0                                              # (B, C, H, W)

        prediction = self.model(xt, t)                                # (B, C, H, W)
        loss = self.criterion(prediction, target)
        return {'loss': loss}
