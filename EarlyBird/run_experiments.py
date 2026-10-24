"""
Run the view_fusion experiments (none | geo | m1) on a single GPU, e.g. an RTX 3060 12GB server.

Same protocol as earlybird-m1-kaggle.ipynb: effective batch 8 (1 GPU x accumulate 8, as configs/t_fit.yml),
50 epochs, test best + last, then a conf_threshold sweep on last. Each run ends with
<out>/results_<arm>_s<seed>.zip, in the same format as the Kaggle notebook.

Run from the EarlyBird/ package dir, ideally inside tmux / nohup:
    python run_experiments.py --data_dir /data/Wildtrack --smoke                 # 20 batches per arm, ~5 min
    python run_experiments.py --data_dir /data/Wildtrack --arms none geo m1 --seeds 42
Re-running the same command after an interruption skips finished steps and resumes training from last.ckpt.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_METRICS = [("MODA", "detect/moda", 1), ("MODP", "detect/modp", 1), ("Precision", "detect/precision", 1),
                 ("Recall", "detect/recall", 1), ("IDF1", "track/idf1", 100), ("MOTA", "track/mota", 100),
                 ("FP", "track/num_false_positives", 1), ("Miss", "track/num_misses", 1),
                 ("ID switches", "track/num_switches", 1)]


def log(msg):
    print(time.strftime("[%m-%d %H:%M:%S] ") + msg, flush=True)


def fmt(v, spec=".4f"):
    return format(v, spec) if v is not None else "-"


class Run:
    def __init__(self, args, arm, seed, smoke=False):
        self.args, self.arm, self.seed, self.smoke = args, arm, seed, smoke
        self.name = f"{'smoke_' if smoke else ''}{arm}_s{seed}"
        self.dir = os.path.abspath(os.path.join(args.out, self.name))
        self.res = os.path.join(self.dir, "results")
        self.ckpt_dir = os.path.join(self.dir, "checkpoints")
        os.makedirs(self.res, exist_ok=True)
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1", "EB_RESULTS_DIR": self.res,
                    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                    "PYTHONWARNINGS": "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"}
        self._write_configs()

    # ------------------------------------------------------------------ configs
    def _write_configs(self):
        data = {"init_args": {"data_dir": self.args.data_dir, "num_workers": self.args.num_workers}}
        progress = {"class_path": "exp_utils.Progress"}
        fit = {"trainer": {
            "accelerator": "gpu", "devices": 1, "max_epochs": self.args.epochs,
            "accumulate_grad_batches": 8, "gradient_clip_val": 0.5, "check_val_every_n_epoch": 1,
            "log_every_n_steps": 5, "enable_progress_bar": False,
            "logger": [
                {"class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                 "init_args": {"save_dir": os.path.join(self.dir, "logs"), "name": "tb"}},
                {"class_path": "lightning.pytorch.loggers.CSVLogger",
                 "init_args": {"save_dir": os.path.join(self.dir, "logs"), "name": "csv"}}],
            "callbacks": [
                {"class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                 "init_args": {"logging_interval": "step"}},
                {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                 "init_args": {"dirpath": self.ckpt_dir, "monitor": "val_loss", "mode": "min", "save_last": True,
                               "filename": "model-{epoch:02d}-{val_loss:.2f}"}},
                progress]},
            "data": data}
        test = {"trainer": {
            "accelerator": "gpu", "devices": 1, "enable_progress_bar": False,
            "logger": {"class_path": "lightning.pytorch.loggers.CSVLogger",
                       "init_args": {"save_dir": os.path.join(self.dir, "logs"), "name": "test"}},
            "callbacks": [progress]},
            "data": data}
        # JSON is valid YAML for jsonargparse
        self.fit_cfg, self.test_cfg = os.path.join(self.dir, "fit.yml"), os.path.join(self.dir, "test.yml")
        json.dump(fit, open(self.fit_cfg, "w"), indent=2)
        json.dump(test, open(self.test_cfg, "w"), indent=2)

    def common(self):
        return ["-c", "configs/d_wildtrack.yml", "-c", f"configs/m_{self.arm}.yml",
                "--seed_everything", str(self.seed)]

    def done(self, step):
        return os.path.exists(os.path.join(self.dir, f".done_{step}"))

    def mark(self, step):
        open(os.path.join(self.dir, f".done_{step}"), "w").close()

    # ------------------------------------------------------------------ main.py subprocess
    def main_py(self, argv, log_name):
        log_path = os.path.join(self.res, log_name)
        cmd = [sys.executable, "main.py", *argv]
        with open(log_path, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            p = subprocess.Popen(cmd, cwd=HERE, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
            for line in p.stdout:
                f.write(line)
                if line.startswith("@@EB "):
                    self._event(json.loads(line[5:]))
            p.wait()
        if p.returncode != 0:
            tail = [l for l in open(log_path).read().splitlines() if not l.startswith("@@EB ")][-30:]
            print("\n".join(tail), flush=True)
            raise RuntimeError(f"{self.name}: main.py exit {p.returncode} – log: {log_path}")

    def _event(self, ev):
        if ev["e"] == "fit_start":
            log(f"  train {ev['world']}x {ev['gpu']} | effective batch {ev['eff_batch']} | {ev['steps']} batch/epoch | "
                f"start epoch {ev['start_epoch']} | view_fusion={ev['view_fusion']} | "
                f"fusion params {ev['fusion_params']:,} | total {ev['params']:,}")
        elif ev["e"] == "epoch_end":
            log(f"  epoch {ev['epoch']:>3}/{self.args.epochs if not self.smoke else 1} | "
                f"train {fmt(ev['train_loss'])} | val {fmt(ev['val_loss'])} | lr {ev['lr']:.2e} | "
                f"{ev['time'] / 60:.1f} min | GPU {ev['mem']:.2f}G (reserved {ev['mem_reserved']:.2f}G) | "
                f"alpha {fmt(ev['alpha'])}")

    # ------------------------------------------------------------------ steps
    def fit(self):
        if self.done("fit"):
            log("  fit: đã xong, bỏ qua")
            return
        argv = ["fit", *self.common(), "-c", self.fit_cfg]
        if self.smoke:
            argv += ["--trainer.max_epochs", "1", "--trainer.limit_train_batches", "20",
                     "--trainer.limit_val_batches", "5"]
        last = os.path.join(self.ckpt_dir, "last.ckpt")
        if os.path.exists(last):
            log(f"  fit: resume từ {last}")
            argv += ["--ckpt_path", last]
        self.main_py(argv, "train.log")
        self.mark("fit")

    def test(self, tag, ckpt):
        """Test ckpt; returns metrics. LightningCLI re-applies the hyper_parameters stored in --ckpt_path."""
        self.main_py(["test", *self.common(), "-c", self.test_cfg, "--ckpt_path", ckpt], f"test_{tag}.log")
        os.replace(os.path.join(self.res, "test_metrics.json"), os.path.join(self.res, f"test_metrics_{tag}.json"))
        pred_dir = sorted(glob.glob(os.path.join(self.dir, "logs", "test", "version_*")), key=os.path.getmtime)[-1]
        os.makedirs(os.path.join(self.res, f"pred_{tag}"), exist_ok=True)
        for f in glob.glob(os.path.join(pred_dir, "*.txt")):
            shutil.copy(f, os.path.join(self.res, f"pred_{tag}"))
        return json.load(open(os.path.join(self.res, f"test_metrics_{tag}.json")))

    def test_best_last(self):
        if self.done("test"):
            log("  test best/last: đã xong, bỏ qua")
            return
        summ = json.load(open(os.path.join(self.res, "train_summary.json")))
        for tag, ckpt in [("best", summ["best_model_path"]), ("last", os.path.join(self.ckpt_dir, "last.ckpt"))]:
            log(f"  test {tag}: {os.path.basename(ckpt)}")
            self.test(tag, ckpt)
        self.mark("test")

    def sweep(self):
        if self.done("sweep"):
            log("  sweep: đã xong, bỏ qua")
            return
        # --model.conf_threshold alone is reset to the trained value by LightningCLI, so the threshold is
        # written into a temporary copy of the checkpoint
        state = torch.load(os.path.join(self.ckpt_dir, "last.ckpt"), map_location="cpu", weights_only=False)
        tmp = os.path.join(self.dir, "sweep_tmp.ckpt")
        out = {}
        for thr in self.args.thresholds:
            t = f"{thr:.2f}"
            state["hyper_parameters"]["conf_threshold"] = thr
            torch.save(state, tmp)
            m = self.test(f"sweep_{t}", tmp)
            out[f"{self.arm}@{t}"] = m
            log(f"  sweep {t}: MODA {m['detect/moda']:.2f} | MOTA {m['track/mota'] * 100:.2f} | "
                f"IDF1 {m['track/idf1'] * 100:.2f} | FP {m['track/num_false_positives']:.0f} | "
                f"Miss {m['track/num_misses']:.0f}")
        os.remove(tmp)
        seen = {(m.get("track/num_false_positives"), m.get("track/num_misses")) for m in out.values()}
        assert len(out) == 1 or len(seen) > 1, f"{self.name}: mọi ngưỡng cho kết quả giống hệt nhau"
        json.dump(out, open(os.path.join(self.res, f"sweep_{self.name}.json"), "w"), indent=2)
        self.mark("sweep")

    def report(self):
        summ = json.load(open(os.path.join(self.res, "train_summary.json")))
        M = {t: json.load(open(os.path.join(self.res, f"test_metrics_{t}.json"))) for t in ["best", "last"]}
        table = {n: {t: (M[t].get(k) * s if M[t].get(k) is not None else None) for t in M}
                 for n, k, s in SWEEP_METRICS}
        raw = json.load(open(os.path.join(self.res, f"sweep_{self.name}.json")))
        sweep = {key.split("@")[1]: {n: (m.get(k) or 0) * s for n, k, s in SWEEP_METRICS} for key, m in raw.items()}
        sweep_mean = {n: sum(v[n] for v in sweep.values()) / len(sweep) for n, _, _ in SWEEP_METRICS}
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        h = summ["history"]
        out = {"run": self.name, "platform": "server",
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "commit": commit,
               "view_fusion": summ["view_fusion"], "fusion_params": summ["fusion_params"],
               "alpha": {t: M[t].get("alpha") for t in M}, "best_ckpt": summ["best_model_path"],
               "train_hours": summ["hours"], "epochs": len(h), "gpu_peak_G": max(x["mem"] for x in h),
               "metrics": table, "sweep_last": sweep, "sweep_last_mean": sweep_mean}
        json.dump(out, open(os.path.join(self.res, f"summary_{self.name}.json"), "w"), indent=2)
        self._curves()

        print(f"\n{self.name}  (view_fusion={out['view_fusion']}, alpha last={fmt(out['alpha']['last'])}, "
              f"{out['train_hours']:.2f} h, GPU peak {out['gpu_peak_G']:.2f} G)")
        print(f"{'Metric':<12} {'best':>8} {'last':>8}")
        for n in ["MODA", "MODP", "Precision", "Recall", "IDF1", "MOTA", "ID switches"]:
            print(f"{n:<12} {fmt(table[n]['best'], '8.2f')} {fmt(table[n]['last'], '8.2f')}")
        print(f"\n{'thr':<6}" + "".join(f"{n:>10}" for n, _, _ in SWEEP_METRICS[:8]))
        for t, v in list(sweep.items()) + [("mean", sweep_mean)]:
            print(f"{t:<6}" + "".join(f"{v[n]:>10.2f}" for n, _, _ in SWEEP_METRICS[:8]))

        zip_base = os.path.join(os.path.abspath(self.args.out), f"results_{self.name}")
        shutil.make_archive(zip_base, "zip", self.res)
        log(f"  📦 {zip_base}.zip")

    def _curves(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rows = []
            for f in sorted(glob.glob(os.path.join(self.dir, "logs", "csv", "version_*", "metrics.csv"))):
                rows += list(csv.DictReader(open(f)))
            num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None
            tr = [(num(r, "step"), num(r, "train_loss")) for r in rows if num(r, "train_loss") is not None]
            va = {int(num(r, "epoch")) + 1: num(r, "val_loss") for r in rows if num(r, "val_loss") is not None}
            h = json.load(open(os.path.join(self.res, "train_summary.json")))["history"]
            al = [(x["epoch"], x["alpha"]) for x in h if x.get("alpha") is not None]
            fig, ax = plt.subplots(1, 3 if al else 2, figsize=(20 if al else 14, 4))
            ax[0].plot(*zip(*tr), alpha=.4)
            ax[0].set(title="Train loss", xlabel="step")
            ax[1].plot(list(va), list(va.values()), marker="o", ms=3)
            ax[1].set(title="Val loss", xlabel="epoch")
            if al:
                ax[2].plot(*zip(*al), marker="o", ms=3)
                ax[2].set(title="M1 gate alpha", xlabel="epoch")
            fig.suptitle(self.name)
            plt.tight_layout()
            fig.savefig(os.path.join(self.res, "curves.png"), dpi=120)
            plt.close(fig)
        except Exception as e:  # plots are optional
            log(f"  (bỏ qua curves.png: {e})")


def check_env(args):
    assert torch.cuda.is_available(), "Không thấy GPU CUDA (torch.version.cuda = %s)" % torch.version.cuda
    p = torch.cuda.get_device_properties(0)
    log(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | GPU 0: {p.name}, {p.total_memory / 2 ** 30:.1f} GB")
    assert "wildtrack" in os.path.basename(os.path.normpath(args.data_dir)).lower(), \
        "--data_dir phải là thư mục tên 'Wildtrack' (datamodule nhận dạng dataset theo tên thư mục)"
    for sub in ["Image_subsets", "annotations_positions", "calibrations"]:
        assert os.path.isdir(os.path.join(args.data_dir, sub)), f"Thiếu {args.data_dir}/{sub}"
    commit = subprocess.run(["git", "-C", HERE, "log", "-1", "--format=%h %s"], capture_output=True, text=True)
    log(f"commit: {commit.stdout.strip()}")


def unit_tests():
    log("unit test (CPU): tests/test_view_fusion.py")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_view_fusion.py", "-q", "-p", "no:warnings"],
                       cwd=HERE, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    assert r.returncode == 0, "Unit test lỗi – dừng, không train"


def overview(args):
    rows = []
    for f in sorted(glob.glob(os.path.join(args.out, "*", "results", "summary_*.json"))):
        s = json.load(open(f))
        if s["run"].startswith("smoke_"):
            continue
        m, sm = s["metrics"], s["sweep_last_mean"]
        rows.append(f"{s['run']:<12} {m['MODA']['last']:>8.2f} {m['MOTA']['last']:>8.2f} {m['IDF1']['last']:>8.2f} "
                    f"{sm['MODA']:>8.2f} {sm['MOTA']:>8.2f} {sm['IDF1']:>8.2f} {fmt(s['alpha']['last'], '8.3f')}")
    if rows:
        print(f"\n{'run':<12} {'MODA':>8} {'MOTA':>8} {'IDF1':>8} {'MODA~':>8} {'MOTA~':>8} {'IDF1~':>8} {'alpha':>8}"
              f"\n{'':<12} {'(last @0.40)':^26} {'(last, mean over thresholds)':^26}")
        print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="thư mục Wildtrack (có Image_subsets, annotations_positions, "
                                                      "calibrations)")
    ap.add_argument("--arms", nargs="+", default=["none", "geo", "m1"], choices=["none", "geo", "m1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.45, 0.50, 0.55, 0.60])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "experiments"))
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 20 batch/arm: kiểm tra chạy được và đo VRAM")
    ap.add_argument("--skip_tests", action="store_true")
    args = ap.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    os.makedirs(args.out, exist_ok=True)

    check_env(args)
    if not args.skip_tests:
        unit_tests()

    if args.smoke:
        res = {}
        for arm in args.arms:
            r = Run(args, arm, args.seeds[0], smoke=True)
            for f in glob.glob(os.path.join(r.dir, ".done_*")) + [os.path.join(r.ckpt_dir, "last.ckpt")]:
                if os.path.exists(f):
                    os.remove(f)
            log(f"===== smoke {arm} =====")
            r.fit()
            h = json.load(open(os.path.join(r.res, "train_summary.json")))["history"][-1]
            res[arm] = h
        print(f"\n{'arm':<5} {'GPU alloc':>10} {'reserved':>10} {'giây (20 batch + val)':>22}")
        for arm, h in res.items():
            print(f"{arm:<5} {h['mem']:>9.2f}G {h['mem_reserved']:>9.2f}G {h['time']:>22.1f}")
        return

    for seed in args.seeds:
        for arm in args.arms:
            r = Run(args, arm, seed)
            log(f"===== {r.name} → {r.dir} =====")
            r.fit()
            r.test_best_last()
            r.sweep()
            r.report()
    overview(args)


if __name__ == "__main__":
    main()
