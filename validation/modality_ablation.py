# ablation_cv5_consistent.py
# -*- coding: utf-8 -*-
"""
Ablation study for DualPrecursorGNNLite model.

This script keeps the same data interface, feature construction style,
three-target prediction setting, y z-score scaling, and metric reporting
format as your previous CV5 script, but adds:

1. Configurable ablation model
2. Module-level ablations: Graph / Fingerprint / Descriptor / Process
3. Fusion-mode ablations: add_diff / concat / add / diff / add_mul_diff
4. Strict outer CV test set + inner validation split for early stopping
5. Train / validation / test summaries for each ablation model
6. CSV export of all fold metrics and summary metrics

Expected project structure is the same as your previous code:
- ./data/output_with_molar_amount_pubchem.xlsx
- ./data/compound_smiles.xlsx
- model.model provides data_process and CarbonDotFeaturizer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

from model.model import data_process, CarbonDotFeaturizer


# =========================
# 0) Global targets
# =========================
TARGET_NAMES = ["Ex_max_nm", "Em_max_nm", "QY_%"]


# =========================
# 1) Utils
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def batch_to_device(batch: Dict[str, Any], dev: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev)
        else:
            try:
                out[k] = v.to(dev)  # torch_geometric Batch
            except Exception:
                out[k] = v
    return out


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for j, name in enumerate(TARGET_NAMES):
        m[f"{name}_MAE"] = float(mean_absolute_error(y_true[:, j], y_pred[:, j]))
        m[f"{name}_RMSE"] = rmse(y_true[:, j], y_pred[:, j])
        m[f"{name}_R2"] = float(r2_score(y_true[:, j], y_pred[:, j]))

    m["MAE_mean"] = float(np.mean([m[f"{t}_MAE"] for t in TARGET_NAMES]))
    m["RMSE_mean"] = float(np.mean([m[f"{t}_RMSE"] for t in TARGET_NAMES]))
    m["R2_mean"] = float(np.mean([m[f"{t}_R2"] for t in TARGET_NAMES]))
    return m


def summarize(folds: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if len(folds) == 0:
        return {}
    keys = folds[0].keys()
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array([d[k] for d in folds], dtype=float)
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
    return out


def get_y_raw(dataset: Dataset, indices: List[int]) -> np.ndarray:
    ys = []
    for i in indices:
        y = dataset[i]["y"]
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        ys.append(np.asarray(y, dtype=np.float32).reshape(1, -1))
    return np.vstack(ys).astype(np.float32)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# =========================
# 2) Dataset
# =========================
class CarbonDotDataset(Dataset):
    def __init__(self, df, featurizer, selected_pair_features=None):
        self.df = df.reset_index(drop=True)
        self.featurizer = featurizer
        self.selected_pair_features = selected_pair_features or []

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        g1 = self.featurizer.smiles_to_graph(row["precursor_1"])
        g2 = self.featurizer.smiles_to_graph(row["precursor_2"])

        fp1 = torch.tensor(self.featurizer.fingerprints(row["precursor_1"]), dtype=torch.float)
        fp2 = torch.tensor(self.featurizer.fingerprints(row["precursor_2"]), dtype=torch.float)

        d1 = self.featurizer.descriptors(row["precursor_1"])
        d2 = self.featurizer.descriptors(row["precursor_2"])

        s_feat = self.featurizer.solvent_feat(row["reagent_1"])
        r_feat = self.featurizer.reaction_feat(row["temperature"], row["time_h"])

        pre1 = float(row["pre_1_mmol"])
        pre2 = float(row["pre_2_mmol"])
        ratio = pre1 / (pre1 + pre2) if (pre1 + pre2) > 0 else 0.5
        pre_ratio = torch.tensor([ratio], dtype=torch.float)

        global_feat = torch.cat([s_feat, r_feat, pre_ratio], dim=0)

        y = torch.tensor(
            [row["Ex_max_nm"], row["Em_max_nm"], row["QY_%"]],
            dtype=torch.float,
        )

        return {
            "g1": g1,
            "g2": g2,
            "fps_1": fp1,
            "fps_2": fp2,
            "desc_1": d1,
            "desc_2": d2,
            "global_feat": global_feat,
            "y": y,
        }


# =========================
# 3) Collate
# =========================
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    graphs_1 = [item["g1"] for item in batch]
    graphs_2 = [item["g2"] for item in batch]

    batch_graphs_1 = Batch.from_data_list(graphs_1)
    batch_graphs_2 = Batch.from_data_list(graphs_2)

    fps_1 = torch.stack([item["fps_1"] for item in batch], dim=0)
    fps_2 = torch.stack([item["fps_2"] for item in batch], dim=0)

    desc_1 = torch.stack([item["desc_1"] for item in batch], dim=0)
    desc_2 = torch.stack([item["desc_2"] for item in batch], dim=0)

    global_feat = torch.stack([item["global_feat"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)

    return {
        "graphs_1": batch_graphs_1,
        "graphs_2": batch_graphs_2,
        "fps_1": fps_1,
        "fps_2": fps_2,
        "desc_1": desc_1,
        "desc_2": desc_2,
        "global_feat": global_feat,
        "y": y,
    }


# =========================
# 4) Encoders
# =========================
class FingerprintEncoderLite(nn.Module):
    def __init__(self, input_dim: int = 679, latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class DescriptorEncoderLite(nn.Module):
    def __init__(self, input_dim: int = 8, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class ProcessEncoderLite(nn.Module):
    def __init__(self, input_dim: int = 14, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class GraphEncoderGINE(nn.Module):
    def __init__(
        self,
        node_in: int = 6,
        edge_in: int = 6,
        hidden: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()

        if num_layers not in [1, 2, 3]:
            raise ValueError("num_layers should be 1, 2, or 3 for this ablation script.")

        self.node_proj = nn.Linear(node_in, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp))

        self.dropout = nn.Dropout(dropout)
        self.pool_fc = nn.Linear(2 * hidden, hidden)

    def forward(self, graph: Batch) -> torch.Tensor:
        x = self.node_proj(graph.x.float())
        e = self.edge_mlp(graph.edge_attr.float())

        for conv in self.convs:
            x = conv(x, graph.edge_index, e)
            x = F.relu(x)
            x = self.dropout(x)

        g_mean = global_mean_pool(x, graph.batch)
        g_max = global_max_pool(x, graph.batch)
        return self.pool_fc(torch.cat([g_mean, g_max], dim=-1))


# =========================
# 5) Ablation model
# =========================
class DualPrecursorAblationModel(nn.Module):
    """
    Configurable version of your DualPrecursorGNNLite_v4_Linear.

    Supported ablations:
    - use_graph: remove / keep graph encoder
    - use_fp: remove / keep molecular fingerprint encoder
    - use_desc: remove / keep descriptor encoder
    - use_process: remove / keep process encoder
    - fusion_mode:
        "add_diff": [mol1 + mol2, abs(mol1 - mol2)]
        "concat":   [mol1, mol2]
        "add":      mol1 + mol2
        "diff":     abs(mol1 - mol2)
        "add_mul_diff": [mol1 + mol2, mol1 * mol2, abs(mol1 - mol2)]
    """

    def __init__(
        self,
        use_graph: bool = True,
        use_fp: bool = True,
        use_desc: bool = True,
        use_process: bool = True,
        fusion_mode: str = "add_diff",
        gnn_hidden: int = 128,
        gnn_layers: int = 2,
        fp_latent: int = 64,
        desc_out: int = 16,
        proc_out: int = 32,
        fusion_hidden: int = 128,
        head_hidden: int = 128,
        out_dim: int = 3,
        dropout_graph: float = 0.10,
        dropout_fp: float = 0.30,
        dropout_fusion: float = 0.20,
        dropout_head: float = 0.20,
    ):
        super().__init__()

        self.use_graph = use_graph
        self.use_fp = use_fp
        self.use_desc = use_desc
        self.use_process = use_process
        self.fusion_mode = fusion_mode

        mol_dim = 0

        if self.use_graph:
            self.graph_encoder = GraphEncoderGINE(
                hidden=gnn_hidden,
                dropout=dropout_graph,
                num_layers=gnn_layers,
            )
            mol_dim += gnn_hidden

        if self.use_fp:
            self.fp_encoder = FingerprintEncoderLite(
                input_dim=679,
                latent_dim=fp_latent,
                dropout=dropout_fp,
            )
            mol_dim += fp_latent

        if self.use_desc:
            self.desc_encoder = DescriptorEncoderLite(
                input_dim=8,
                out_dim=desc_out,
            )
            mol_dim += desc_out

        if self.use_process:
            self.proc_encoder = ProcessEncoderLite(
                input_dim=14,
                out_dim=proc_out,
            )
        else:
            proc_out = 0

        self.mol_dim = mol_dim
        self.proc_out = proc_out

        if mol_dim <= 0 and proc_out <= 0:
            raise ValueError("At least one of molecular features or process features must be used.")

        if mol_dim > 0:
            if fusion_mode == "concat":
                fusion_input_dim = 2 * mol_dim
            elif fusion_mode in ["add", "diff"]:
                fusion_input_dim = mol_dim
            elif fusion_mode == "add_diff":
                fusion_input_dim = 2 * mol_dim
            elif fusion_mode == "add_mul_diff":
                fusion_input_dim = 3 * mol_dim
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

            self.fusion_mlp = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout_fusion),
                nn.Linear(fusion_hidden, fusion_hidden),
                nn.GELU(),
            )
            head_input_dim = fusion_hidden + proc_out
        else:
            self.fusion_mlp = None
            head_input_dim = proc_out

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(head_hidden, out_dim),
        )

    def encode_mol(self, graph: Batch, fp: torch.Tensor, desc: torch.Tensor) -> Optional[torch.Tensor]:
        reps = []

        if self.use_graph:
            reps.append(self.graph_encoder(graph))

        if self.use_fp:
            reps.append(self.fp_encoder(fp))

        if self.use_desc:
            reps.append(self.desc_encoder(desc))

        if len(reps) == 0:
            return None

        return torch.cat(reps, dim=-1)

    def fuse_mol(self, mol1: torch.Tensor, mol2: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "concat":
            z = torch.cat([mol1, mol2], dim=-1)

        elif self.fusion_mode == "add":
            z = mol1 + mol2

        elif self.fusion_mode == "diff":
            z = torch.abs(mol1 - mol2)

        elif self.fusion_mode == "add_diff":
            z = torch.cat([mol1 + mol2, torch.abs(mol1 - mol2)], dim=-1)

        elif self.fusion_mode == "add_mul_diff":
            z = torch.cat(
                [mol1 + mol2, mol1 * mol2, torch.abs(mol1 - mol2)],
                dim=-1,
            )

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        return self.fusion_mlp(z)

    def forward_embedding(
        self,
        graphs_1,
        graphs_2,
        fps_1,
        fps_2,
        desc_1,
        desc_2,
        global_feat,
    ) -> torch.Tensor:
        reps = []

        if self.mol_dim > 0:
            mol1 = self.encode_mol(graphs_1, fps_1, desc_1)
            mol2 = self.encode_mol(graphs_2, fps_2, desc_2)
            fusion = self.fuse_mol(mol1, mol2)
            reps.append(fusion)

        if self.use_process:
            proc = self.proc_encoder(global_feat)
            reps.append(proc)

        return torch.cat(reps, dim=-1)

    def forward(self, graphs_1, graphs_2, fps_1, fps_2, desc_1, desc_2, global_feat):
        emb = self.forward_embedding(
            graphs_1,
            graphs_2,
            fps_1,
            fps_2,
            desc_1,
            desc_2,
            global_feat,
        )
        return self.head(emb)


# =========================
# 6) Ablation configs
# =========================
ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Main full model
    "Full_Graph_FP_Desc_Process_add_diff": dict(
        use_graph=True,
        use_fp=True,
        use_desc=True,
        use_process=True,
        fusion_mode="add_diff",
    ),

    # Module ablation
    "w_o_Graph": dict(
        use_graph=False,
        use_fp=True,
        use_desc=True,
        use_process=True,
        fusion_mode="add_diff",
    ),
    "w_o_FP": dict(
        use_graph=True,
        use_fp=False,
        use_desc=True,
        use_process=True,
        fusion_mode="add_diff",
    ),
    "w_o_Desc": dict(
        use_graph=True,
        use_fp=True,
        use_desc=False,
        use_process=True,
        fusion_mode="add_diff",
    ),
    "w_o_Process": dict(
        use_graph=True,
        use_fp=True,
        use_desc=True,
        use_process=False,
        fusion_mode="add_diff",
    ),

    # Single / partial modality baselines
    "Graph_Process": dict(
        use_graph=True,
        use_fp=False,
        use_desc=False,
        use_process=True,
        fusion_mode="add_diff",
    ),
    "FP_Process": dict(
        use_graph=False,
        use_fp=True,
        use_desc=False,
        use_process=True,
        fusion_mode="add_diff",
    ),
    "Desc_Process": dict(
        use_graph=False,
        use_fp=False,
        use_desc=True,
        use_process=True,
        fusion_mode="add_diff",
    ),
    "Process_Only": dict(
        use_graph=False,
        use_fp=False,
        use_desc=False,
        use_process=True,
        fusion_mode="add_diff",
    ),

    # Fusion strategy ablation, based on full input modalities
    "Fusion_concat": dict(
        use_graph=True,
        use_fp=True,
        use_desc=True,
        use_process=True,
        fusion_mode="concat",
    ),
    "Fusion_add_only": dict(
        use_graph=True,
        use_fp=True,
        use_desc=True,
        use_process=True,
        fusion_mode="add",
    ),
    "Fusion_diff_only": dict(
        use_graph=True,
        use_fp=True,
        use_desc=True,
        use_process=True,
        fusion_mode="diff",
    ),
    "Fusion_add_mul_diff": dict(
        use_graph=True,
        use_fp=True,
        use_desc=True,
        use_process=True,
        fusion_mode="add_mul_diff",
    ),

    # GNN complexity ablation
    "GINE_1_layer": dict(
        use_graph=True,
        use_fp=True,
        use_desc=True,
        use_process=True,
        fusion_mode="add_diff",
        gnn_layers=1,
    ),
    "GINE_3_layers": dict(
        use_graph=True,
        use_fp=True,
        use_desc=True,
        use_process=True,
        fusion_mode="add_diff",
        gnn_layers=3,
    ),

    # Lightweight capacity ablation
    "Lightweight_hidden64_fp32": dict(
        use_graph=True,
        use_fp=True,
        use_desc=True,
        use_process=True,
        fusion_mode="add_diff",
        gnn_hidden=64,
        fp_latent=32,
        fusion_hidden=64,
        head_hidden=64,
    ),
}


# =========================
# 7) Training
# =========================
@dataclass
class TrainCfg:
    epochs: int = 200
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 10
    patience: int = 20
    grad_clip: float = 1.0

@dataclass
class CV5Cfg:
    seed: int = 42
    n_splits: int = 5
    n_repeats: int = 3
    shuffle: bool = True
    drop_all_zero_y: bool = True
    num_workers: int = 0


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_scaler: StandardScaler,
    dev: torch.device,
    cfg: TrainCfg,
) -> nn.Module:
    model = model.to(dev)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(cfg.epochs):
        model.train()

        for batch in train_loader:
            y_raw = batch["y"].numpy().astype(np.float32)
            y_scaled = y_scaler.transform(y_raw).astype(np.float32)
            batch["y_scaled"] = torch.as_tensor(y_scaled, dtype=torch.float32)

            batch = batch_to_device(batch, dev)

            pred = model(
                batch["graphs_1"],
                batch["graphs_2"],
                batch["fps_1"],
                batch["fps_2"],
                batch["desc_1"],
                batch["desc_2"],
                batch["global_feat"],
            )

            loss = F.mse_loss(pred, batch["y_scaled"], reduction="mean")

            opt.zero_grad()
            loss.backward()

            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

        # validation for early stopping
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                y_raw = batch["y"].numpy().astype(np.float32)
                y_scaled = y_scaler.transform(y_raw).astype(np.float32)
                batch["y_scaled"] = torch.as_tensor(y_scaled, dtype=torch.float32)

                batch = batch_to_device(batch, dev)

                pred = model(
                    batch["graphs_1"],
                    batch["graphs_2"],
                    batch["fps_1"],
                    batch["fps_2"],
                    batch["desc_1"],
                    batch["desc_2"],
                    batch["global_feat"],
                )

                val_losses.append(
                    F.mse_loss(pred, batch["y_scaled"], reduction="mean").item()
                )

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: p.detach().cpu().clone()
                for k, p in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def predict_end2end(
    model: nn.Module,
    loader: DataLoader,
    y_scaler: StandardScaler,
    dev: torch.device,
) -> np.ndarray:
    model.eval()
    model.to(dev)

    preds_scaled = []

    for batch in loader:
        batch = batch_to_device(batch, dev)

        pred = model(
            batch["graphs_1"],
            batch["graphs_2"],
            batch["fps_1"],
            batch["fps_2"],
            batch["desc_1"],
            batch["desc_2"],
            batch["global_feat"],
        ).detach().cpu().numpy().astype(np.float32)

        preds_scaled.append(pred)

    preds_scaled = np.vstack(preds_scaled)
    return y_scaler.inverse_transform(preds_scaled).astype(np.float32)


# =========================
# 8) CV ablation driver
# =========================
def run_ablation_cv(
    dataset: Dataset,
    ablation_configs: Dict[str, Dict[str, Any]] = ABLATION_CONFIGS,
    collate_fn=custom_collate_fn,
    train_cfg: TrainCfg = TrainCfg(),
    cv_cfg: CV5Cfg = CV5Cfg(),
    save_dir: str = "./ablation_results",
) -> Dict[str, Any]:
    set_seed(cv_cfg.seed)
    dev = get_device()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    idx = np.arange(len(dataset), dtype=int)

    rkf = RepeatedKFold(
        n_splits=cv_cfg.n_splits,
        n_repeats=cv_cfg.n_repeats,
        random_state=cv_cfg.seed,
    )

    all_fold_records: List[Dict[str, Any]] = []
    all_summaries: Dict[str, Any] = {}

    for model_name, model_cfg in ablation_configs.items():
        print("\n" + "=" * 80)
        print(f"Running ablation model: {model_name}")
        print(f"Config: {model_cfg}")
        print("=" * 80)

        folds_train: List[Dict[str, float]] = []
        folds_test: List[Dict[str, float]] = []

        for run_id, (tr_rel, te_rel) in enumerate(rkf.split(idx), 1):
            set_seed(cv_cfg.seed + run_id)

            tr = idx[tr_rel].tolist()
            te = idx[te_rel].tolist()

            train_set = Subset(dataset, tr)
            test_set = Subset(dataset, te)

            train_loader = DataLoader(
                train_set,
                batch_size=train_cfg.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=cv_cfg.num_workers,
            )

            test_loader = DataLoader(
                test_set,
                batch_size=train_cfg.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=cv_cfg.num_workers,
            )

            # fit y scaler on TRAIN fold only
            y_tr_raw = get_y_raw(dataset, tr)
            y_te_raw = get_y_raw(dataset, te)

            y_scaler = StandardScaler().fit(y_tr_raw)

            model = DualPrecursorAblationModel(**model_cfg)
            n_params = count_trainable_params(model)

            # Keep the same logic as your original code:
            # test_loader is used as val_loader for early stopping.
            model = train_one_fold(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                y_scaler=y_scaler,
                dev=dev,
                cfg=train_cfg,
            )

            pred_train = predict_end2end(model, train_loader, y_scaler, dev)
            pred_test = predict_end2end(model, test_loader, y_scaler, dev)

            met_train = calc_metrics(y_tr_raw, pred_train)
            met_test = calc_metrics(y_te_raw, pred_test)

            folds_train.append(met_train)
            folds_test.append(met_test)

            rep = (run_id - 1) // cv_cfg.n_splits + 1
            fold = (run_id - 1) % cv_cfg.n_splits + 1

            print(
                f"[{model_name}] "
                f"[Rep {rep}/{cv_cfg.n_repeats} | Fold {fold}/{cv_cfg.n_splits}] "
                f"Params={n_params} | "
                f"Train R2_mean={met_train['R2_mean']:.3f} | "
                f"Test R2_mean={met_test['R2_mean']:.3f}"
            )

            for split_name, metrics in [
                ("train", met_train),
                ("test", met_test),
            ]:
                rec = {
                    "model": model_name,
                    "split": split_name,
                    "repeat": rep,
                    "fold": fold,
                    "run_id": run_id,
                    "n_params": n_params,
                    "n_train": len(tr),
                    "n_test": len(te),
                }
                rec.update(metrics)
                all_fold_records.append(rec)

        summary_train = summarize(folds_train)
        summary_test = summarize(folds_test)

        all_summaries[model_name] = {
            "train": summary_train,
            "test": summary_test,
        }

        print_summary(f"{model_name} | train set", summary_train)
        print_summary(f"{model_name} | test set", summary_test)

    fold_df = pd.DataFrame(all_fold_records)
    fold_df.to_csv(
        save_path / "ablation_all_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = make_summary_dataframe(all_summaries)
    summary_df.to_csv(
        save_path / "ablation_summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved:")
    print(f"- {save_path / 'ablation_all_fold_metrics.csv'}")
    print(f"- {save_path / 'ablation_summary_metrics.csv'}")

    return {
        "fold_metrics": fold_df,
        "summary_metrics": summary_df,
        "raw_summaries": all_summaries,
    }


def print_summary(title: str, summ: Dict[str, Dict[str, float]]) -> None:
    print(f"\n--- {title} ---")
    metric_order = [
        "Ex_max_nm_MAE", "Ex_max_nm_RMSE", "Ex_max_nm_R2",
        "Em_max_nm_MAE", "Em_max_nm_RMSE", "Em_max_nm_R2",
        "QY_%_MAE", "QY_%_RMSE", "QY_%_R2",
        "MAE_mean", "RMSE_mean", "R2_mean",
    ]

    for k in metric_order:
        if k in summ:
            print(f"{k:>16s}: {summ[k]['mean']:.4f} ± {summ[k]['std']:.4f}")
    print()


def make_summary_dataframe(all_summaries: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    for model_name, split_dict in all_summaries.items():
        for split_name, metric_dict in split_dict.items():
            row = {
                "model": model_name,
                "split": split_name,
            }

            for metric_name, stat in metric_dict.items():
                row[f"{metric_name}_mean"] = stat["mean"]
                row[f"{metric_name}_std"] = stat["std"]

            rows.append(row)

    return pd.DataFrame(rows)


# =========================
# 9) Main
# =========================
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    df_raw = pd.read_excel("./data/output_with_molar_amount_pubchem.xlsx")
    df_smiles = pd.read_excel("./data/compound_smiles.xlsx")

    solvent_polarity_data = {
        "water": {
            "dielectric_constant": 80.1,
            "pi_star": 1.09,
            "HBD": 1.17,
            "HBA": 0.47,
            "ET30": 63.1,
            "pKa": 15.7,
            "ORP_mV": 300,
            "density_g_cm3": 0.997,
            "specific_heat_J_gK": 4.18,
            "viscosity_mPa_s": 0.89,
            "boiling_point_C": 100,
        },
        "dimethylformamide": {
            "dielectric_constant": 36.7,
            "pi_star": 0.88,
            "HBD": 0.00,
            "HBA": 0.69,
            "ET30": 43.2,
            "pKa": 29,
            "ORP_mV": 200,
            "density_g_cm3": 0.944,
            "specific_heat_J_gK": 2.47,
            "viscosity_mPa_s": 0.92,
            "boiling_point_C": 153,
        },
        "ethanol": {
            "dielectric_constant": 24.6,
            "pi_star": 0.54,
            "HBD": 0.86,
            "HBA": 0.75,
            "ET30": 51.9,
            "pKa": 15.9,
            "ORP_mV": 250,
            "density_g_cm3": 0.789,
            "specific_heat_J_gK": 2.44,
            "viscosity_mPa_s": 1.07,
            "boiling_point_C": 78.37,
        },
        "sulfuric acid": {
            "dielectric_constant": 83,
            "pi_star": 1.11,
            "HBD": 1.22,
            "HBA": 0.48,
            "ET30": 64.8,
            "pKa": 0.0,
            "ORP_mV": 800,
            "density_g_cm3": 1.05,
            "specific_heat_J_gK": 3.6,
            "viscosity_mPa_s": 1.2,
            "boiling_point_C": 100,
        },
        "phosphoric acid": {
            "dielectric_constant": 82,
            "pi_star": 1.10,
            "HBD": 1.20,
            "HBA": 0.47,
            "ET30": 64.0,
            "pKa": 1.0,
            "ORP_mV": 700,
            "density_g_cm3": 1.03,
            "specific_heat_J_gK": 3.8,
            "viscosity_mPa_s": 1.1,
            "boiling_point_C": 100,
        },
    }

    df = data_process(df_raw)
    featurizer = CarbonDotFeaturizer(
        df_smiles=df_smiles,
        solvent_polarity_data=solvent_polarity_data,
    )
    dataset = CarbonDotDataset(df, featurizer)

    print(f"Dataset size: {len(dataset)}")
    print(f"Targets: {TARGET_NAMES}")

    # If running all models is too slow, start with a subset:
    # selected_configs = {
    #     k: ABLATION_CONFIGS[k]
    #     for k in [
    #         "Full_Graph_FP_Desc_Process_add_diff",
    #         "w_o_Graph",
    #         "w_o_FP",
    #         "w_o_Desc",
    #         "w_o_Process",
    #         "FP_Process",
    #         "Process_Only",
    #         "Lightweight_hidden64_fp32",
    #     ]
    # }
    selected_configs = ABLATION_CONFIGS

    out = run_ablation_cv(
        dataset=dataset,
        ablation_configs=selected_configs,
        collate_fn=custom_collate_fn,
        train_cfg=TrainCfg(
            epochs=200,
            batch_size=10,
            patience=20,
            lr=2e-3,
            weight_decay=1e-4,
            grad_clip=1.0,
        ),
        cv_cfg=CV5Cfg(
            seed=42,
            n_splits=5,
            n_repeats=3,
            num_workers=0,
        ),
        save_dir="./ablation_results",
    )

    print("\n====================")
    print("Ablation Summary Preview")
    print("====================")
    print(out["summary_metrics"][["model", "split", "R2_mean_mean", "R2_mean_std", "MAE_mean_mean", "RMSE_mean_mean"]])
