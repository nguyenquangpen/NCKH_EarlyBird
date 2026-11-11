"""Progress callback for the view_fusion experiments (run_experiments.py).

Emits '@@EB {json}' events on stdout (epoch table, memory, M1 alpha) and writes
train_summary.json / test_metrics.json to $EB_RESULTS_DIR. Does not touch training.
The epoch history is part of the callback state, so it survives a resume from last.ckpt.
"""
import json
import math
import os
import time

import torch
from lightning.pytorch.callbacks import Callback


def _out_dir():
    return os.environ.get("EB_RESULTS_DIR", "results")


def _f(x):
    try:
        v = float(x.item() if hasattr(x, "item") else x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def emit(kind, **d):
    print("@@EB " + json.dumps({"e": kind, **d}), flush=True)


def _fusion_info(pl_module):
    model = pl_module.model
    fusion = getattr(model, "fusion", None)
    return {"view_fusion": getattr(model, "view_fusion", "none"),
            "alpha": _f(fusion.alpha) if fusion is not None and fusion.mode == "m1" else None,
            "fusion_params": sum(p.numel() for p in fusion.parameters()) if fusion is not None else 0}


class Progress(Callback):
    def __init__(self):
        super().__init__()
        self.t0 = self.ep_t0 = time.time()
        self.losses, self.history = [], []

    def state_dict(self):
        return {"history": self.history}

    def load_state_dict(self, state_dict):
        self.history = list(state_dict.get("history", []))

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.t0 = time.time()
        bs = getattr(trainer.datamodule, "batch_size", 1)
        emit("fit_start", max_epochs=trainer.max_epochs, start_epoch=trainer.current_epoch,
             steps=trainer.num_training_batches, world=trainer.world_size,
             eff_batch=bs * trainer.accumulate_grad_batches * trainer.world_size,
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
             params=sum(p.numel() for p in pl_module.parameters()), **_fusion_info(pl_module))

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.ep_t0, self.losses = time.time(), []
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.is_global_zero:
            v = _f(trainer.callback_metrics.get("train_loss"))
            if v is not None:
                self.losses.append(v)

    def on_train_epoch_end(self, trainer, pl_module):  # Lightning 2.x: runs after validation
        if not trainer.is_global_zero:
            return
        cm = trainer.callback_metrics
        row = {"epoch": trainer.current_epoch + 1,
               "train_loss": sum(self.losses) / len(self.losses) if self.losses else None,
               "val_loss": _f(cm.get("val_loss")),
               "lr": trainer.optimizers[0].param_groups[0]["lr"],
               "time": time.time() - self.ep_t0,
               "mem": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0,
               "mem_reserved": torch.cuda.max_memory_reserved() / 2 ** 30 if torch.cuda.is_available() else 0,
               "alpha": _fusion_info(pl_module)["alpha"]}
        self.history.append(row)
        emit("epoch_end", **row)

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        ck = trainer.checkpoint_callback
        s = {"best_model_path": ck.best_model_path, "best_val_loss": _f(ck.best_model_score),
             "last_model_path": ck.last_model_path,
             "hours": sum(h["time"] for h in self.history) / 3600,
             **_fusion_info(pl_module), "history": self.history}
        os.makedirs(_out_dir(), exist_ok=True)
        json.dump(s, open(os.path.join(_out_dir(), "train_summary.json"), "w"), indent=2)
        emit("fit_end", **{k: v for k, v in s.items() if k != "history"})

    def on_test_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            m = {k: _f(v) for k, v in trainer.callback_metrics.items()}
            m.update(_fusion_info(pl_module))
            os.makedirs(_out_dir(), exist_ok=True)
            json.dump(m, open(os.path.join(_out_dir(), "test_metrics.json"), "w"), indent=2)
            emit("test_end")
