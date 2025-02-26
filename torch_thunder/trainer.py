import os
import time
from typing import Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from torch._dynamo import OptimizedModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_thunder.module import ThunderModule

_verbose = False


def logging(
    message: str,
):
    # TODO: Add logging to file

    if _verbose:
        print(message)


def get_model_params(model: nn.Module) -> str:
    """
    Returns the number of parameters in the model in a human-readable format

    Args:
        model (nn.Module): Model to get the number of parameters

    Returns:
        str: Human-readable format of the number of parameters
    """
    num_params = sum(p.numel() for p in model.parameters())
    if num_params < 1e3:
        fmt_num_params = f"{num_params:.0f}"
    elif num_params < 1e6:
        fmt_num_params = f"{num_params/1e3:.2f}K"
    elif num_params < 1e9:
        fmt_num_params = f"{num_params/1e6:.2f}M"
    else:
        fmt_num_params = f"{num_params/1e9:.2f}B"
    return fmt_num_params


def _save_loss_history(
    loss_history: dict[str, list[float]],
    ckpt_dir: str,
    epoch: int,
    val_interval: int,
    save_csv: bool = True,
    abort_on_nan: bool = True,
):
    """
    Save the loss history plot and the data as a csv (optional)

    Args:
        loss_history (dict[str, list[float]]): Loss history data
        ckpt_dir (str): Checkpoint directory
        epoch (int): Current epoch
        val_interval (int): Validation interval
        save_csv (bool): Save not only the plot but also the data as a csv.
            Defaults to True.
        abort_on_nan (bool): Abort if the loss history contains NaN or Inf values.
    """
    assert (
        len(loss_history["train"]) == epoch + 1
    ), "Invalid length for training loss history"
    assert (
        len(loss_history["val"]) == (epoch // val_interval) + 1
    ), "Invalid length for validation loss history"
    if abort_on_nan:
        loss_list = np.array([loss_history["train"], loss_history["val"]])
        if np.isnan(loss_list).any() or np.isinf(loss_list).any():
            raise ValueError("Loss history contains NaN or Inf values")

    fig, ax_train = plt.subplots()
    c = "tab:blue"
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train loss", color=c)
    ax_train.plot(list(range(epoch + 1)), loss_history["train"], label="train", color=c)
    ax_train.grid(visible=True)
    ax_train.legend(loc="upper center")
    ax_train.tick_params(axis="y", labelcolor=c)

    ax_val = ax_train.twinx()
    c = "tab:orange"
    ax_val.set_ylabel("Validation loss", color=c)
    ax_val.plot(
        list(range(0, epoch + 1, val_interval)),
        loss_history["val"],
        label="validation",
        color=c,
    )
    ax_val.grid(None)
    ax_val.legend(loc="upper right")
    ax_val.tick_params(axis="y", labelcolor=c)

    plt.title("Loss History")
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/loss_history.png")
    plt.close()

    if save_csv:
        with open(f"{ckpt_dir}/loss_history.csv", "w") as f:
            f.write("epoch,train,val\n")
            for i in range(epoch):
                val_loss = (
                    loss_history["val"][i // val_interval]
                    if i + 1 != epoch
                    else loss_history["val"][-1]
                )
                f.write(f"{i+1},{loss_history['train'][i]:.6f},{val_loss:.6f}\n")


def thunder_train(
    model: ThunderModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    optimizer: Optimizer,
    ckpt_basedir: str = "checkpoints",
    ckpt_subdir: Optional[str] = None,
    ckpt_name_format: str = "epoch={epoch}_val={val_loss}_train={train_loss}.ckpt",
    ckpt_save_interval: int = 1,
    clip_grad_value: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    scheduler: Optional[LRScheduler] = None,
    compile_model: bool = True,
    compile_options: Optional[dict] = None,
    device: Literal["cuda", "cpu", "mps"] = "cuda",
    use_amp: bool = False,
    exist_ok: bool = False,
    save_last_ckpt: bool = True,
    save_trainer_args: bool = True,
    save_loss_history: bool = True,
    save_loss_csv: bool = False,
    abort_on_nan: bool = True,
    log_max_vram_usage: bool = False,
    verbose: bool = True,
) -> Optional[str]:
    """
    Thunder training loop

    Args:
        model (ThunderModule): Model to train
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        n_epochs (int): Number of epochs to train.
        optimizer (Optimizer): Optimizer for training.
        ckpt_basedir (str, optional): Checkpoint base directory.
            Defaults to "checkpoints".
        ckpt_subdir (Optional[str], optional): Checkpoint subdirectory.
            Full path will be `ckpt_basedir/ckpt_subdir`.
            Defaults to None.
        ckpt_name_format (str, optional): Checkpoint name format.
            Allowed keys: {epoch}, {val_loss}, {train_loss}.
            Defaults to "epoch={epoch}_val={val_loss}_train={train_loss}.ckpt".
        ckpt_save_interval (int, optional): Checkpoint save interval.
            Defaults to 1.
        clip_grad_value (Optional[float], optional): Clip gradient value.
            Defaults to None.
        clip_grad_norm (Optional[float], optional): Clip gradient norm.
            Defaults to None.
        scheduler (Optional[LRScheduler], optional): Learning rate scheduler.
            Defaults to None.
        compile_model (bool, optional): Compile model.
            Defaults to True.
        compile_options (Optional[dict], optional): Compilation options.
            Defaults to None.
        device (Literal["cuda", "cpu", "mps"], optional): Device to use.
            Defaults to "cuda".
        use_amp (bool, optional): Use Automatic Mixed Precision.
            Defaults to False.
        exist_ok (bool, optional): Overwrite the existing checkpoint directory.
            Defaults to False.
        save_last_ckpt (bool, optional): Save the last checkpoint.
            Defaults to True.
            Otherwise, the last checkpoint will be removed and
            only the best checkpoint will be saved.
        save_trainer_args (bool, optional): Save trainer arguments.
            Defaults to True.
        save_loss_history (bool, optional): Save loss history plot.
            Defaults to True.
        save_loss_csv (bool, optional): Save loss history as not only a plot but also a csv.
            Defaults to False.
        abort_on_nan (bool, optional): Abort if the loss history contains NaN or Inf values.
            Defaults to True.
        log_max_vram_usage (bool, optional): Log the maximum VRAM usage in GB.
            Defaults to False.
        verbose (bool, optional): Verbose output.
            Defaults to True.

    Returns:
        str: Checkpoint directory where the model was saved.
    """
    global _verbose
    _verbose = verbose

    logging("[i] The number of model parameters: " + get_model_params(model))

    assert isinstance(
        model, (ThunderModule, OptimizedModule)
    ), "Model should be an instance of ThunderModule"

    assert (
        clip_grad_value is None or clip_grad_norm is None
    ), "Either clip_grad_value or clip_grad_norm can be set, not both"

    ckpt_dir = f"{ckpt_basedir}/{ckpt_subdir}/" if ckpt_subdir else f"{ckpt_basedir}/"
    if os.path.exists(ckpt_dir):
        exist_flag = False
        for f in os.listdir(ckpt_dir):
            if f.endswith(".ckpt"):
                exist_flag = True
                break
        if exist_flag and not exist_ok:
            logging(
                f"[!] Checkpoint directory '{ckpt_dir.rstrip('/')}' already exists. Aborted."
            )
            return None
        elif exist_flag and exist_ok:
            logging(
                f"[!] Checkpoint directory '{ckpt_dir.rstrip('/')}' already exists. Overwriting."
            )
    else:
        logging(f"[i] Checkpoint directory: {ckpt_dir.rstrip('/')}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model_class_name = model.__class__.__name__
    is_model_compiled = isinstance(model, OptimizedModule)
    if is_model_compiled:
        model_class_name = getattr(model, "_orig_mod").__class__.__name__

    if compile_model and is_model_compiled:
        logging("[i] Model is already compiled; skipping compilation.")
    elif compile_model and not is_model_compiled:
        assert hasattr(torch, "compile"), "[!] Couldn't find `torch.compile`"
        try:
            model: ThunderModule = torch.compile(  # type: ignore
                model,
                **(compile_options or {}),
            )
        except Exception as e:
            logging(f"[!] Model compilation failed: {e}")
            return None
        else:
            logging("[i] Model compiled successfully")

    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available"
    elif device == "mps":
        assert hasattr(torch.backends, "mps"), "MPS is not available"
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging(
                    "[!] MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                logging(
                    "[!] MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
    torch_device: torch.device = torch.device(device)

    if torch_device.type == "cuda":
        _name = torch.cuda.get_device_name(0)
        _ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging(f"[i] Using CUDA device ({_name}, {_ram:.2f}GB VRAM)")
    elif torch_device.type == "mps":
        logging("[i] Using MPS backend")
    else:
        logging("[!] Using CPU backend")
    model.to(torch_device)

    if save_trainer_args:
        with open(f"{ckpt_dir}/trainer_hparams.txt", "w") as f:
            f.write(
                f"model: {model_class_name}\n"
                f"params: {get_model_params(model)}\n"
                f"epochs: {n_epochs}\n"
                f"optimizer: {optimizer.__class__.__name__}\n"
                f"ckpt_save_interval: {ckpt_save_interval}\n"
                f"clip_grad_value: {clip_grad_value}\n"
                f"clip_grad_norm: {clip_grad_norm}\n"
                f"scheduler: {scheduler.__class__.__name__ if scheduler else None}\n"
                f"compile_model: {compile_model}\n"
                f"compile_options: {compile_options}\n"
                f"device: {device}\n"
                f"use_amp: {use_amp}\n"
            )

    amp_grad_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    loss_history = {
        "train": [],
        "val": [],
    }

    epoch_bar = tqdm(
        range(0, n_epochs),
        desc="Epoch",
        ascii=True,
        position=0,
    )

    _prev_last_ckpt_name: str = ""
    _prev_best_ckpt_name: str = ""

    max_vram_usage: float = 0.0  # in GB

    start_time = time.time()

    """Training loop"""
    for epoch in epoch_bar:
        epoch_bar.set_description(f"Epoch {epoch}")
        model.train()
        _train_loss_minibatch: list[float] = []

        batch: tuple[Tensor, ...]
        for batch_idx, batch in enumerate(
            batch_bar := tqdm(
                train_loader,
                desc="Batch",
                ascii=True,
                leave=epoch == n_epochs - 1,
                position=1,
            ),
        ):
            batch = tuple(
                (t.to(torch_device) if isinstance(t, Tensor) else t) for t in batch
            )
            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type=torch_device.type):
                    loss = model.training_step(batch, batch_idx=batch_idx, epoch=epoch)
                amp_grad_scaler.scale(loss).backward()
            else:
                loss = model.training_step(batch, batch_idx=batch_idx, epoch=epoch)
                loss.backward()

            _train_loss_minibatch.append(float(loss.item()))

            # If using AMP, unscale the gradients before clipping
            if use_amp and (clip_grad_value is not None or clip_grad_norm is not None):
                amp_grad_scaler.unscale_(optimizer)

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            if clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

            if use_amp:
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            batch_bar.set_postfix(
                loss=f"{loss.item():.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

        loss_history["train"].append(np.mean(_train_loss_minibatch))
        if np.isnan(loss_history["train"][-1]):
            tqdm.write("[!] Training loss contains NaN values!")
        if np.isinf(loss_history["train"][-1]):
            tqdm.write("[!] Training loss contains Inf values!")

        """
        Validation/Save step

        Condition 1: Every ckpt_save_interval epochs
        Condition 2: When the last epoch
        """
        if (epoch % ckpt_save_interval == 0) or (epoch == n_epochs - 1):
            model.eval()
            _val_loss_history: list[float] = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(
                    val_bar := tqdm(
                        val_loader,
                        desc="Validation",
                        ascii=True,
                        leave=False,
                        position=1,
                    ),
                ):
                    batch = tuple(
                        (t.to(torch_device) if isinstance(t, Tensor) else t)
                        for t in batch
                    )
                    loss = model.validation_step(
                        batch, batch_idx=batch_idx, epoch=epoch
                    ).item()

                    _val_loss_history.append(float(loss))
                    val_bar.set_postfix(loss=f"{loss:.6f}")

            loss_history["val"].append(np.mean(_val_loss_history))

            """Save the last checkpoint"""
            if save_last_ckpt:
                if _prev_last_ckpt_name:
                    os.remove(f"{ckpt_dir}/{_prev_last_ckpt_name}")
                _prev_last_ckpt_name = "last_" + ckpt_name_format.format(
                    epoch=epoch,
                    val_loss=f"{loss_history['val'][-1]:.6f}",
                    train_loss=f"{loss_history['train'][-1]:.6f}",
                )
                torch.save(model.state_dict(), f"{ckpt_dir}/{_prev_last_ckpt_name}")

            """Save the best checkpoint"""
            if loss_history["val"][-1] == min(loss_history["val"]):
                if _prev_best_ckpt_name:
                    os.remove(f"{ckpt_dir}/{_prev_best_ckpt_name}")
                _prev_best_ckpt_name = "best_" + ckpt_name_format.format(
                    epoch=epoch,
                    val_loss=f"{loss_history['val'][-1]:.6f}",
                    train_loss=f"{loss_history['train'][-1]:.6f}",
                )
                torch.save(model.state_dict(), f"{ckpt_dir}/{_prev_best_ckpt_name}")

            """Save loss history"""
            if save_loss_history:
                _save_loss_history(
                    loss_history,
                    ckpt_dir,
                    epoch,
                    val_interval=ckpt_save_interval,
                    save_csv=save_loss_csv,
                    abort_on_nan=abort_on_nan,
                )

            if log_max_vram_usage and torch_device.type == "cuda":
                gfree, total = torch.cuda.mem_get_info(torch_device.index)
                max_vram_usage = max(max_vram_usage, (total - gfree) / 1024**3)

        epoch_bar.set_postfix(
            train_loss=f"{loss_history['train'][-1]:.6f}",
            val_loss=f"{loss_history['val'][-1]:.6f}",
        )

    end_time = time.time()
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

    with open(f"{ckpt_dir}/trainer_hparams.txt", "a") as f:
        f.write("[i] Training completed in " + total_time_str + "\n")
        if log_max_vram_usage and torch_device.type == "cuda":
            f.write(f"max_vram_usage: {max_vram_usage:.2f}GB\n")
            logging(f"[i] Maximum VRAM usage: {max_vram_usage:.2f}GB")

    logging("[i] Training completed in " + total_time_str)

    return ckpt_dir
