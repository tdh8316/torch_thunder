import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch_thunder as tt

hparams = {
    "hidden_size": 512,
    "batch_size": 256,
    "lr": 1e-3,
    "epochs": 5,
}


class MLP(tt.ThunderModule):
    """
    MLP model for MNIST classification
    """

    def __init__(self, args: dict):
        super(MLP, self).__init__()

        self.hidden_size = args["hidden_size"]
        self.dropout = args["dropout"]

        self.layers = nn.Sequential(
            nn.Linear(28 * 28, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 10),
        )

    def hyperparameters(self) -> dict:
        return hparams

    def loss_fn(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        return torch.nn.functional.cross_entropy(pred, target)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, 28 * 28)
        return self.layers(x)

    def training_step(self, batch: tuple[Tensor, ...], **_) -> tt.Loss:
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        return loss

    def validation_step(self, batch: tuple[Tensor, ...], **_) -> tt.Loss:
        x, y = batch
        pred = self.prediction_step(x)
        loss = self.loss_fn(pred, y)
        return loss

    def prediction_step(self, x: Tensor) -> Tensor:
        pred = self(x)
        return pred


def main():
    from torchvision import datasets, transforms

    train_dataset = datasets.MNIST(
        root="../data/mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    val_dataset = datasets.MNIST(
        root="../data/mnist",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
    )

    model = MLP(hparams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"])

    ckpt_dir = tt.thunder_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=hparams["epochs"],
        ckpt_subdir="mlp_mnist",
        device="cpu",
    )

    if ckpt_dir:
        tt.save_hparams_from_dict(f"{ckpt_dir}/hparams.json", hparams)


if __name__ == "__main__":
    main()
