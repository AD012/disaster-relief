"""Fine-tune Grounding DINO Tiny on SARD train tiles (Phase 5). Head-only or LoRA + head.

Usage:
    python src/train.py --config configs/p5_head.yaml                      # fresh run
    python src/train.py --config configs/p5_head.yaml --resume auto        # resume if a checkpoint exists
    python src/train.py --config configs/p5_lora.yaml --output-dir runs/p5-lora-try2 --resume auto   # under $DR_ROOT
    python src/train.py --config configs/p5_head.yaml --dry-run            # 20 train / 20 val tiles, 1 epoch
    python src/train.py --config configs/p5_head.yaml --set train.lr=5e-5  # override a config value

Data: reads ONLY the train and val manifests. Calib and test are never opened.
Selection: early stopping and every hyperparameter choice use val AP50 (COCO, IoU 0.5,
ignore regions as in evaluate.py). Epoch 0 = the untrained model, scored the same way.

Head: the parameters named in cfg["head"] (default: the shared box MLP and the text
projection; the class head is a parameter-free text/image dot product). Trained fully.
LoRA (mode: lora): low-rank adapters on the decoder layers matched by cfg["lora"]["targets"].
The box MLP is tied across all decoder levels by shared parameters, so it is trained as a
head parameter, not wrapped by PEFT (PEFT would give each tied copy its own adapter).

Loss: the model's own Hungarian-matched terms, re-weighted with cfg["loss_weights"]. The
encoder (two-stage) class logits of the released checkpoint are unscaled (about -140..60),
so the *_enc terms default to weight 0.

Paths: data_root, output_dir and the HF cache resolve under the project root DR_ROOT (env var;
default: the repo's parent folder), so the same configs work on the Mac and the HPC:
    $DR_ROOT/data/      raw tiles (cfg["data_root"])        $DR_ROOT/runs/   checkpoints, exports, logs
    $DR_ROOT/hf_cache/  HF_HOME, if the folder exists        $DR_ROOT/logs/   Slurm job logs
    $DR_ROOT/repo/      this repo on the HPC (any name on the Mac)
Manifests and results/metrics stay inside the repo (versioned, frozen splits come from git).

Outputs in output_dir (default $DR_ROOT/runs/<run_name>):
    checkpoints/last.pt        full training state, written atomically every epoch (and on
                               SIGTERM/SIGUSR1 at the next optimizer step, then exit 75)
    export/last/, export/best/ weights in the format src/detector.py --adapter loads:
                               head mode: export/<x>/head.pt (state dict)
                               lora mode: export/<x>/ (adapter_config.json + LoRA weights + head.pt)
    log.csv, run.json          per-epoch train/val loss and val AP50; commit, dirty flag, config hash
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from utils import dr_root, git_info, in_repo, load_config, set_seed, under_root   # no transformers import

if (dr_root() / "hf_cache").is_dir():   # must be set before transformers / huggingface_hub are imported
    os.environ.setdefault("HF_HOME", str(dr_root() / "hf_cache"))

import argparse, contextlib, csv, hashlib, io, json, math, random, re, shutil, signal, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from detector import peft_compat
from evaluate import coco_stats

ALLOWED_SPLITS = ("train", "val")   # calib is for conformal calibration, test is used once at the end
EXIT_PREEMPTED = 75                 # checkpoint written, job should be requeued
# batch 1: a training step needs ~6.9 GB on MPS at batch 1 and ~12 GB at batch 2, which swaps on a 16 GB Mac
DRY_RUN = {"train_subset": 20, "val_subset": 20, "train.epochs": 1, "train.batch_size": 1,
           "train.num_workers": 0, "eval.batch_size": 2}


# ---------------------------------------------------------------- config

def set_key(cfg, dotted, value):
    *path, last = dotted.split(".")
    node = cfg
    for k in path:
        node = node.setdefault(k, {})
    node[last] = value


def config_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k != "output_dir"}   # moving the output dir is not a new run
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def autocast(dev, precision):
    if dev != "cuda" or precision == "fp32":
        return contextlib.nullcontext()   # MPS and CPU always run fp32
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


# ---------------------------------------------------------------- data

def load_split(cfg, split):
    if split not in ALLOWED_SPLITS:
        raise SystemExit(f"train.py only reads {ALLOWED_SPLITS}, not {split!r}")
    return in_repo(cfg["manifests"]) / f"{split}.json"


class Tiles(Dataset):
    """Tiles from one manifest. Ignore regions (iscrowd, min side < 4 px) are left out of the targets."""

    def __init__(self, manifest, data_root, subset, seed, hflip=False):
        coco = json.loads(Path(manifest).read_text())
        images = sorted(coco["images"], key=lambda im: im["id"])
        if subset:
            images = sorted(random.Random(seed).sample(images, subset), key=lambda im: im["id"])
        self.images, self.root, self.seed, self.hflip, self.epoch = images, under_root(data_root), seed, hflip, 0
        self.anns = defaultdict(list)
        for a in coco["annotations"]:
            if not a.get("iscrowd") and a["bbox"][2] > 0 and a["bbox"][3] > 0:
                self.anns[a["image_id"]].append(a["bbox"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = Image.open(self.root / im["file_name"]).convert("RGB")
        W, H = img.size
        boxes = torch.tensor([[(x + w / 2) / W, (y + h / 2) / H, w / W, h / H] for x, y, w, h in self.anns[im["id"]]],
                             dtype=torch.float32).reshape(-1, 4)
        # flip decision depends only on (seed, epoch, image id), so a resumed epoch sees the same data
        if self.hflip and random.Random(f"{self.seed}-{self.epoch}-{im['id']}").random() < 0.5:
            img = ImageOps.mirror(img)
            boxes[:, 0] = 1 - boxes[:, 0]
        return img, {"class_labels": torch.zeros(len(boxes), dtype=torch.long), "boxes": boxes}, im["id"]


class Collate:
    def __init__(self, proc, prompt, scale):
        self.proc, self.prompt = proc, prompt
        self.size = {"shortest_edge": round(800 * scale), "longest_edge": round(1333 * scale)}   # as baseline.detect

    def __call__(self, batch):
        imgs, targets, ids = zip(*batch)
        inputs = self.proc(images=list(imgs), text=[self.prompt] * len(imgs), return_tensors="pt", size=self.size)
        return inputs, list(targets), list(ids), [im.size[::-1] for im in imgs]


# ---------------------------------------------------------------- model

def build(cfg, dev):
    """Model with only the head (and LoRA) trainable. Returns (model, proc, head) where head is
    [(name, param)] with names relative to GroundingDinoForObjectDetection, as head.pt stores them."""
    proc = AutoProcessor.from_pretrained(cfg["model"])
    net = GroundingDinoForObjectDetection.from_pretrained(cfg["model"])
    net.requires_grad_(False)
    prefixes = cfg["head"]
    head = [(n, p) for n, p in net.named_parameters() if any(n == q or n.startswith(q + ".") for q in prefixes)]
    unmatched = [q for q in prefixes if not any(n == q or n.startswith(q + ".") for n, _ in head)]
    if unmatched:
        raise SystemExit(f"head prefixes matched no parameters: {unmatched}")
    model = net
    if cfg["mode"] == "lora":
        from peft import LoraConfig, get_peft_model
        peft_compat()
        lc = cfg["lora"]
        target = "(" + "|".join(lc["targets"]) + ")"
        overlap = [n for n, _ in net.named_modules() if re.fullmatch(target, n)
                   and any(n.startswith(q) for q in prefixes)]
        if overlap:
            raise SystemExit(f"LoRA targets overlap the head: {overlap[:3]}")
        model = get_peft_model(net, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                               target_modules=target, bias="none"))
    elif cfg["mode"] != "head":
        raise SystemExit(f"mode must be head or lora, not {cfg['mode']!r}")
    for _, p in head:
        p.requires_grad_(True)
    return model.to(dev), proc, head


def train_mode(model, cfg):
    """Frozen parts stay in eval mode (no dropout noise; MPS attention has no dropout kernel).
    Only the decoder, which holds the LoRA adapters, trains with dropout; the head MLP has none."""
    model.eval()
    if cfg["mode"] == "lora":
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        base.model.decoder.train()


def param_groups(model, head, cfg):
    head_ids = {id(p) for _, p in head}
    lora = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    t = cfg["train"]
    groups = [{"params": [p for _, p in head], "lr": t["lr"]}]
    if lora:
        groups.append({"params": lora, "lr": t["lr_lora"]})
    return groups


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def weighted_loss(loss_dict, weights):
    total = sum(w * loss_dict[k] for k, w in weights.items() if w and k in loss_dict)
    return total, {k: loss_dict[k].detach().item() for k in weights if k in loss_dict}


def export(model, head, cfg, out_dir, meta):
    """Write weights in the format detector.load_adapter reads; swap in atomically-ish (tmp dir + rename)."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    torch.save({n: p.detach().cpu().clone() for n, p in head}, tmp / "head.pt")
    if cfg["mode"] == "lora":
        model.save_pretrained(tmp)   # adapter_config.json + LoRA weights only (head params are not PEFT-managed)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    return str(out_dir / "head.pt") if cfg["mode"] == "head" else str(out_dir)


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate_val(model, loader, gt, cfg, dev, precision):
    """Val loss and COCO AP on the val tiles; detections are kept and rounded exactly as baseline.py does."""
    model.eval()
    dets, sums, n = [], defaultdict(float), 0
    proc = loader.collate_fn.proc
    for inputs, targets, ids, sizes in loader:
        inputs = inputs.to(dev)
        with autocast(dev, precision):
            out = model(**inputs, labels=[{k: v.to(dev) for k, v in t.items()} for t in targets])
        total, parts = weighted_loss(out.loss_dict, cfg["loss_weights"])
        sums["loss"] += float(total) * len(ids)
        for k, v in parts.items():
            sums[k] += v * len(ids)
        n += len(ids)
        out.logits, out.pred_boxes = out.logits.float(), out.pred_boxes.float()
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=cfg["eval"]["floor"], text_threshold=cfg["eval"]["text_threshold"],
            target_sizes=sizes)
        for image_id, r in zip(ids, res):
            for (x0, y0, x1, y1), s in zip(r["boxes"].tolist(), r["scores"].tolist()):
                dets.append({"image_id": image_id, "category_id": 1, "score": round(s, 4),
                             "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]})
    stats = coco_stats(gt, dets, [im["id"] for im in loader.dataset.images])
    return {"val_loss": sums.pop("loss") / n, **{f"val_{k}": v / n for k, v in sums.items()},
            "val_AP50": stats["AP50"], "val_AP50:95": stats["AP50:95"], "val_boxes": len(dets)}


# ---------------------------------------------------------------- checkpointing and logs

class Preempt:
    """SIGTERM / SIGUSR1 (Slurm preemption or --signal) -> checkpoint at the next optimizer step and exit."""

    def __init__(self, simulate_after=None):
        self.flag, self.simulate_after, self.steps = False, simulate_after, 0
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, self._set)

    def _set(self, signum, frame):
        print(f"signal {signum}: will checkpoint at the next optimizer step", flush=True)
        self.flag = True

    def step(self):
        self.steps += 1
        if self.simulate_after and self.steps >= self.simulate_after:
            self.flag = True
        return self.flag


def atomic_save(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def write_csv(path, history):
    keys = list(dict.fromkeys(k for row in history for k in row))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
    os.replace(tmp, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


# ---------------------------------------------------------------- main

def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", help="overrides output_dir (relative = under $DR_ROOT)")
    ap.add_argument("--resume", default="none", help="none | auto | path/to/last.pt")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. train.lr=5e-5")
    ap.add_argument("--dry-run", action="store_true", help="20 train / 20 val tiles, 1 epoch (CI on the Mac)")
    ap.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes (recorded in run.json)")
    ap.add_argument("--simulate-preempt", type=int, metavar="N", help="testing: act as if signalled after N steps")
    return ap.parse_args()


def main():
    args = parse()
    cfg = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_key(cfg, k, yaml.safe_load(v))
    if args.dry_run:
        for k, v in DRY_RUN.items():
            set_key(cfg, k, v)
        cfg["run_name"] += "-dryrun"
        cfg["dry_run"] = True
    if not under_root(cfg["data_root"]).is_dir():
        raise SystemExit(f"data_root {under_root(cfg['data_root'])} not found; set DR_ROOT (now {dr_root()}) "
                         f"to the folder that holds data/ (default: the repo's parent folder)")
    out = under_root(args.output_dir or cfg.get("output_dir") or Path("runs") / cfg["run_name"])
    cfg["output_dir"] = str(out)
    chash = config_hash(cfg)

    git = git_info()
    if not git["commit"] or git["dirty"]:
        msg = "no git checkout found" if not git["commit"] else "uncommitted changes in tracked files"
        if not (args.allow_dirty or args.dry_run):
            raise SystemExit(f"{msg}; commit first (run.json must record dirty=false) or pass --allow-dirty")
        print(f"warning: {msg}; recorded in run.json")

    ckpt_path = out / "checkpoints" / "last.pt"
    resume_from = None
    if args.resume == "auto":
        resume_from = ckpt_path if ckpt_path.exists() else None
    elif args.resume != "none":
        resume_from = Path(args.resume)
    elif ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} exists; pass --resume auto to continue it or choose another --output-dir")
    state = torch.load(resume_from, map_location="cpu", weights_only=False) if resume_from else None
    if state and state["config_hash"] != chash:
        raise SystemExit(f"config changed since {resume_from} was written (hash {state['config_hash'][:12]} "
                         f"vs {chash[:12]}); use a new --output-dir")
    if state and state["status"] == "done":
        print(f"{out}: already finished (best epoch {state['best']['epoch']}, val AP50 {state['best']['val_AP50']})")
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    set_seed(cfg["seed"])
    dev = pick_device()
    precision = cfg["train"]["cuda_precision"] if dev == "cuda" else "fp32"
    model, proc, head = build(cfg, dev)
    t = cfg["train"]

    train_ds = Tiles(load_split(cfg, "train"), cfg["data_root"], cfg.get("train_subset"), cfg["seed"], t["hflip"])
    val_ds = Tiles(load_split(cfg, "val"), cfg["data_root"], cfg.get("val_subset"), cfg["seed"])
    collate = Collate(proc, cfg["prompt"], cfg["scale"])
    val_loader = DataLoader(val_ds, batch_size=cfg["eval"]["batch_size"], collate_fn=collate,
                            num_workers=t["num_workers"])
    with contextlib.redirect_stdout(io.StringIO()):
        val_gt = COCO(str(load_split(cfg, "val")))

    batches_per_epoch = math.ceil(len(train_ds) / t["batch_size"])
    updates_per_epoch = math.ceil(batches_per_epoch / t["grad_accum"])
    total_updates = updates_per_epoch * t["epochs"]
    opt = torch.optim.AdamW(param_groups(model, head, cfg), weight_decay=t["weight_decay"])
    warm = t["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_updates - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    n_head = sum(p.numel() for _, p in head)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    run_json = out / "run.json"
    if state:
        named = dict(model.named_parameters())
        for n, v in state["trainable"].items():
            named[n].data.copy_(v.to(dev))
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"]); set_rng_state(state["rng"])
        run = json.loads(run_json.read_text())
        run["resumes"].append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git,
                               "slurm_job": os.environ.get("SLURM_JOB_ID"), "epoch": state["epoch"],
                               "step": state["step"]})
        if git["commit"] != run["git"]["commit"]:
            print(f"warning: resuming at commit {git['commit'][:8]}, run started at {run['git']['commit'][:8]}")
        print(f"resumed from {resume_from}: epoch {state['epoch']}, step {state['step']}/{batches_per_epoch}")
    else:
        import peft, transformers
        state = {"epoch": 1, "step": 0, "history": [], "best": {"epoch": None, "val_AP50": -1.0},
                 "bad_epochs": 0, "running": {}, "status": "running"}
        run = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git,
               "config_hash": chash, "config": args.config, "device": dev, "precision": precision,
               "paths": {"dr_root": str(dr_root()), "data_root": str(train_ds.root), "output_dir": str(out),
                         "hf_home": os.environ.get("HF_HOME")},
               "slurm_job": os.environ.get("SLURM_JOB_ID"),
               "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                            "transformers": transformers.__version__, "peft": peft.__version__},
               "tiles": {"train": len(train_ds), "val": len(val_ds)}, "params": {"head": n_head, "trainable": n_train},
               "resumes": []}
    print(f"{cfg['run_name']}: mode {cfg['mode']}, {dev}/{precision}, {len(train_ds)} train / {len(val_ds)} val tiles, "
          f"{n_train:,} trainable params ({n_head:,} head)", flush=True)

    def finish_epoch(epoch, row):
        state["history"].append(row)
        meta = {"epoch": epoch, "val_AP50": row["val_AP50"], "config_hash": chash, "commit": git["commit"],
                "mode": cfg["mode"], "model": cfg["model"]}
        run["export_last"] = export(model, head, cfg, out / "export" / "last", meta)
        bar = -math.inf if state["best"]["epoch"] is None else state["best"]["val_AP50"] + cfg["early_stopping"]["min_delta"]
        if row["val_AP50"] > bar:
            state["best"] = {"epoch": epoch, "val_AP50": row["val_AP50"]}
            state["bad_epochs"] = 0
            run["export_best"] = export(model, head, cfg, out / "export" / "best", meta)
        elif epoch:
            state["bad_epochs"] += 1
        row["best"] = state["best"]["epoch"] == epoch

    def save(status):
        state.update(status=status, trainable=trainable_state(model), optimizer=opt.state_dict(),
                     scheduler=sched.state_dict(), scaler=scaler.state_dict(), rng=rng_state(), config_hash=chash)
        atomic_save(state, ckpt_path)
        run.update(status=status, best=state["best"], history=state["history"],
                   updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        write_json(run_json, run)
        write_csv(out / "log.csv", state["history"])

    if not state["history"] and cfg["eval"]["before_training"]:
        t0 = time.time()
        row = {"epoch": 0, **evaluate_val(model, val_loader, val_gt, cfg, dev, precision), "train_loss": None,
               "lr": None, "seconds": round(time.time() - t0, 1)}
        finish_epoch(0, row)
        save("running")
        print(f"epoch 0 (untrained): val AP50 {row['val_AP50']:.4f}, val loss {row['val_loss']:.4f}", flush=True)

    preempt = Preempt(args.simulate_preempt)
    while state["epoch"] <= t["epochs"]:
        epoch = state["epoch"]
        train_ds.epoch = epoch
        order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(cfg["seed"] * 1000 + epoch)).tolist()
        batches = [order[i:i + t["batch_size"]] for i in range(0, len(order), t["batch_size"])]
        loader = DataLoader(train_ds, batch_sampler=batches[state["step"]:], collate_fn=collate,
                            num_workers=t["num_workers"])
        train_mode(model, cfg)
        run_sums = defaultdict(float, state["running"])
        t0 = time.time()
        for k, (inputs, targets, ids, _) in enumerate(loader, state["step"] + 1):
            inputs = inputs.to(dev)
            with autocast(dev, precision):
                out_ = model(**inputs, labels=[{kk: v.to(dev) for kk, v in tt.items()} for tt in targets])
            loss, parts = weighted_loss(out_.loss_dict, cfg["loss_weights"])
            scaler.scale(loss / t["grad_accum"]).backward()
            run_sums["loss"] += loss.detach().item() * len(ids)
            run_sums["n"] += len(ids)
            for kk, v in parts.items():
                run_sums[kk] += v * len(ids)
            if k % t["grad_accum"] == 0 or k == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], t["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                if preempt.step() and k < len(batches):
                    state["step"], state["running"] = k, dict(run_sums)
                    save("preempted")
                    print(f"checkpointed at epoch {epoch} step {k}/{len(batches)}; exit {EXIT_PREEMPTED}", flush=True)
                    sys.exit(EXIT_PREEMPTED)
            if k % 50 == 0:
                print(f"epoch {epoch} step {k}/{len(batches)} loss {run_sums['loss'] / run_sums['n']:.4f}", flush=True)
        train_secs = time.time() - t0
        n = run_sums.pop("n")
        row = {"epoch": epoch, "train_loss": run_sums.pop("loss") / n,
               **{f"train_{kk}": v / n for kk, v in run_sums.items()}, "lr": opt.param_groups[0]["lr"]}
        row.update(evaluate_val(model, val_loader, val_gt, cfg, dev, precision))
        row["seconds"] = round(time.time() - t0, 1)
        row["train_s_per_tile"] = round(train_secs / max(1, len(batches) - state["step"]) / t["batch_size"], 3)
        finish_epoch(epoch, row)
        state.update(epoch=epoch + 1, step=0, running={})
        stop = state["bad_epochs"] >= cfg["early_stopping"]["patience"]
        done = stop or epoch == t["epochs"]
        save("done" if done else "running")
        print(f"epoch {epoch}: train loss {row['train_loss']:.4f}, val loss {row['val_loss']:.4f}, "
              f"val AP50 {row['val_AP50']:.4f}{' (best)' if row['best'] else ''}, {row['seconds']} s", flush=True)
        if stop:
            print(f"early stop: no val AP50 gain > {cfg['early_stopping']['min_delta']} for "
                  f"{cfg['early_stopping']['patience']} epochs")
        if done:
            break
        if preempt.flag:
            save("preempted")
            sys.exit(EXIT_PREEMPTED)

    print(f"best: epoch {state['best']['epoch']}, val AP50 {state['best']['val_AP50']:.4f} -> {run.get('export_best')}")
    if not cfg.get("dry_run"):
        in_repo("results/metrics").mkdir(parents=True, exist_ok=True)
        write_json(in_repo("results/metrics") / f"{cfg['run_name']}-train.json",
                   {k: run[k] for k in ("git", "config_hash", "config", "device", "precision", "tiles", "params",
                                        "best", "history")})


if __name__ == "__main__":
    main()
