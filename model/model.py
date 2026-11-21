# cd_dualgnn.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_add_pool

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import Crippen, rdMolDescriptors


# ---------------------------
# 0) 通用：随机种子 & device
# ---------------------------
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------
# 1) 数据预处理（与你原来的归一化一致）
# ---------------------------
def min_max_normalize(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    series = series.clip(lower=min_val, upper=max_val)
    return series.apply(lambda x: (x - min_val) / (max_val - min_val))


def data_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    对应你 notebook 中 data_process：
    - dropna(Ex_max_nm, Em_max_nm, stoke_shift, QY_%)
    - Em_max_nm: 400~650
    - QY_%: 0~100
    - stoke_shift: 40~180
    """
    df_new = df.dropna(subset=["Ex_max_nm", "Em_max_nm", "Stokes_shift_nm", "QY_%"]).reset_index(drop=True)
    #df_new["Em_max_nm"] = min_max_normalize(df_new["Em_max_nm"], 400, 650)
    #df_new["QY_%"] = min_max_normalize(df_new["QY_%"], 0, 100)
    #df_new["stoke_shift"] = min_max_normalize(df_new["stoke_shift"], 40, 180)
    return df_new


# ---------------------------
# 2) 化学表示：SMILES -> 图；指纹；描述符；溶剂/条件归一化
# ---------------------------
def _fingerprint_to_numpy(fp) -> np.ndarray:
    if hasattr(fp, "GetNumBits"):
        arr = np.zeros((fp.GetNumBits(),), dtype=int)
        ConvertToNumpyArray(fp, arr)
        return arr
    if hasattr(fp, "ToBitVect"):
        bitvect = fp.ToBitVect()
        arr = np.zeros((bitvect.GetNumBits(),), dtype=int)
        ConvertToNumpyArray(bitvect, arr)
        return arr
    raise ValueError("Unsupported fingerprint type")


@dataclass
class FeaturizerConfig:
    # 指纹维度：你当前用 MACCS(167) + Avalon(nBits=512) => 679
    avalon_nbits: int = 512

    # 描述符归一化尺度（保持与你 notebook 一致）
    desc_div: Tuple[float, ...] = (150.0, 150.0, 50.0, 10.0, 3.0, 10.0, 10.0, 100.0)

    # 反应条件归一化范围（保持与你 notebook 一致）
    temp_min: float = 160.0
    temp_max: float = 220.0
    time_min: float = 4.0
    time_max: float = 8.0

    # 溶剂特征 min/max（你 notebook 是基于 5 种溶剂/酸的范围）
    solvent_min: Tuple[float, ...] = (24.6, 0.54, 0.00, 0.47, 43.2, 0.0, 200.0, 0.789, 2.44, 0.89, 78.37)
    solvent_max: Tuple[float, ...] = (83.0, 1.11, 1.22, 0.75, 64.8, 29.0, 800.0, 1.05, 4.18, 1.20, 153.0)


class CarbonDotFeaturizer:
    """
    你 notebook 中的：
    - smiles_to_graph
    - get_all_fingerprints_as_numpy
    - get_molecular_descriptors + normalize_desc
    - normalize_reaction_feat
    - normalize_solvent_feat
    统一封装。
    """

    def __init__(
        self,
        df_smiles: pd.DataFrame,
        solvent_polarity_data: Dict[str, Dict[str, float]],
        cfg: FeaturizerConfig = FeaturizerConfig(),
    ):
        self.df_smiles = df_smiles.copy()
        self.df_smiles["compound_name"] = self.df_smiles["compound_name"].astype(str)
        self.solvent_polarity_data = {str(k).strip().lower(): v for k, v in solvent_polarity_data.items()}
        self.cfg = cfg

    def _get_smiles(self, name: str) -> str:
        name = str(name)
        hit = self.df_smiles.loc[self.df_smiles["compound_name"] == name, "SMILES"]
        if hit.empty:
            raise ValueError(f"Compound name '{name}' not found in df_smiles")
        return hit.values[0]

    def smiles_to_graph(self, name: str) -> Data:
        smiles = self._get_smiles(name)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit cannot parse SMILES for '{name}': {smiles}")

        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization().real,
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
            ])
        x = torch.tensor(atom_features, dtype=torch.float)

        edges, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            bond_feats = [
                float(bt == Chem.rdchem.BondType.SINGLE),
                float(bt == Chem.rdchem.BondType.DOUBLE),
                float(bt == Chem.rdchem.BondType.TRIPLE),
                float(bt == Chem.rdchem.BondType.AROMATIC),
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
            ]
            edges += [(i, j), (j, i)]
            edge_attrs += [bond_feats, bond_feats]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def fingerprints(self, name: str) -> np.ndarray:
        smiles = self._get_smiles(name)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit cannot parse SMILES for '{name}'")

        maccs_fp = MACCSkeys.GenMACCSKeys(mol)            # 167
        avalon_fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.cfg.avalon_nbits)  # 512
        combined = np.concatenate([_fingerprint_to_numpy(maccs_fp), _fingerprint_to_numpy(avalon_fp)])
        return combined.astype(np.float32)  # [679]

    def descriptors(self, name: str) -> torch.Tensor:
        smiles = self._get_smiles(name)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit cannot parse SMILES for '{name}'")

        # 你 notebook 的 8 维描述符
        MolarRefractivity = Crippen.MolMR(mol)
        TPSA = rdMolDescriptors.CalcTPSA(mol)
        HeavyAtomCount = rdMolDescriptors.CalcNumHeavyAtoms(mol)
        Complexity = rdMolDescriptors.CalcChi0v(mol)
        AromaticRings = rdMolDescriptors.CalcNumAromaticRings(mol)
        Complexity_Chi0v = rdMolDescriptors.CalcChi0v(mol)

        conjugated_double_bonds = 0
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0 and bond.GetIsConjugated():
                conjugated_double_bonds += 1

        w1, w2, w3 = 1.0, 1.0, 0.05
        ConjugationScore = w1 * conjugated_double_bonds + w2 * AromaticRings + w3 * TPSA

        desc = torch.tensor(
            [MolarRefractivity, TPSA, HeavyAtomCount, Complexity, AromaticRings,
             Complexity_Chi0v, conjugated_double_bonds, ConjugationScore],
            dtype=torch.float,
        )
        div = torch.tensor(self.cfg.desc_div, dtype=torch.float)
        return desc / div  # normalize_desc

    def reaction_feat(self, temperature: float, time_h: float) -> torch.Tensor:
        t_norm = (float(temperature) - self.cfg.temp_min) / (self.cfg.temp_max - self.cfg.temp_min)
        time_norm = (float(time_h) - self.cfg.time_min) / (self.cfg.time_max - self.cfg.time_min)
        return torch.tensor([t_norm, time_norm], dtype=torch.float)

    def solvent_feat(self, reagent_1: str) -> torch.Tensor:
        key = str(reagent_1).strip().lower()
        if key not in self.solvent_polarity_data:
            raise KeyError(f"Solvent '{reagent_1}' not found in solvent_polarity_data")
        vals = list(self.solvent_polarity_data[key].values())
        x = torch.tensor(vals, dtype=torch.float)

        min_vals = torch.tensor(self.cfg.solvent_min, dtype=torch.float)
        max_vals = torch.tensor(self.cfg.solvent_max, dtype=torch.float)
        return (x - min_vals) / (max_vals - min_vals)


# ---------------------------
# 3) Dataset + collate
# ---------------------------
class CarbonDotDataset(Dataset):
    """
    返回结构与你 notebook 一致：
    graphs(2), fps_1/2, desc_1/2, global_feat(溶剂11+条件2+比例1=14), y(3)
    """

    def __init__(self, df: pd.DataFrame, featurizer: CarbonDotFeaturizer):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.featurizer = featurizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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

        global_feat = torch.cat([s_feat, r_feat, pre_ratio], dim=0)  # 14维

        #y = torch.tensor([row["Em_max_nm"], row["stoke_shift"], row["QY_%"]], dtype=torch.float)
        #y = torch.tensor([row["Ex_max_nm"], row["Em_max_nm"],row["QY_%"]], dtype=torch.float)
        y = torch.tensor([row["Ex_max_nm"], row["Em_max_nm"]], dtype=torch.float)

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


def build_loaders(
    df: pd.DataFrame,
    featurizer: CarbonDotFeaturizer,
    batch_size: int = 10,
    test_size: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    dataset = CarbonDotDataset(df, featurizer)
    idx = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=custom_collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=num_workers
    )
    return train_loader, test_loader


# ---------------------------
# 4) 模型：DualPrecursorGNN()
# ---------------------------
class FingerprintAutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 679, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DualPrecursorGNN(nn.Module):
    """
    结构保持与你 notebook 相同：
    - edge_embedding(6->hidden)
    - GATv2Conv 3层（第一层输入6；后两层残差）
    - fp_encoder(679->128)
    - desc_fc(8->32)
    - cross_att(embed=hidden+128+32, heads=4)
    - condition_fc(14->hidden)
    - regressor([attn_mean + cond] -> 3 + Sigmoid)
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        fingerprint_dim: int = 679,
        descriptor_dim: int = 8,
        condition_dim: int = 14,
        attn_heads: int = 4,
        node_dropout_p: float = 0.1,
        reg_dropout_p: float = 0.3,
    ):
        super().__init__()

        self.edge_embedding = nn.Linear(6, hidden_dim)

        self.gnn_layers = nn.ModuleList([
            GATv2Conv(6, hidden_dim, edge_dim=hidden_dim),
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim),
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim),
        ])
        self.node_dropout = nn.Dropout(p=node_dropout_p)

        self.fp_encoder = FingerprintAutoEncoder(input_dim=fingerprint_dim, latent_dim=128)

        self.desc_fc = nn.Sequential(
            nn.Linear(descriptor_dim, 32),
            nn.GELU(),
        )

        self.cross_att = nn.MultiheadAttention(
            embed_dim=hidden_dim + 128 + 32,
            num_heads=attn_heads,
            batch_first=True,
        )

        self.condition_fc = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, hidden_dim),
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 512),   # attn_mean(hidden) + cond(hidden)
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=reg_dropout_p),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=reg_dropout_p),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )

    def encode_graph(self, graph: Batch) -> torch.Tensor:
        x = graph.x.float()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr.float())

        for i, conv in enumerate(self.gnn_layers):
            if i == 0:
                x = conv(x, edge_index, edge_attr)
                x = F.gelu(x)
                x = self.node_dropout(x)
            else:
                residual = x
                x = conv(x, edge_index, edge_attr)
                x = F.gelu(x)
                x = self.node_dropout(x)
                x = x + residual

        return global_add_pool(x, graph.batch)

    def forward(
        self,
        graphs_1: Batch,
        graphs_2: Batch,
        fps_1: torch.Tensor,
        fps_2: torch.Tensor,
        desc_1: torch.Tensor,
        desc_2: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g1 = self.encode_graph(graphs_1)
        g2 = self.encode_graph(graphs_2)

        fp1 = self.fp_encoder(fps_1)
        fp2 = self.fp_encoder(fps_2)

        d1 = self.desc_fc(desc_1)
        d2 = self.desc_fc(desc_2)

        mol1 = torch.cat([g1, fp1, d1], dim=-1)
        mol2 = torch.cat([g2, fp2, d2], dim=-1)

        attn_in = torch.stack([mol1, mol2], dim=1)              # [B,2,E]
        attn_out, attn_w = self.cross_att(attn_in, attn_in, attn_in, need_weights=True)
        attn_out = attn_out.mean(dim=1)                         # [B,E]

        cond = self.condition_fc(global_feat)                   # [B,hidden]
        combined = torch.cat([attn_out[:, :cond.size(1)], cond], dim=-1) \
            if attn_out.size(1) != cond.size(1) else torch.cat([attn_out, cond], dim=-1)

        # 注意：你的 notebook 是把 attn_out(=E维)直接 concat cond(hidden) 再 regressor，
        # 这里为了保持 regressor 输入为 2*hidden，我们取 attn_out 前 hidden 维。
        # 如果你想完全一致，也可以把 regressor 第一层改为 Linear(E+hidden, 512)。

        out = self.regressor(combined)
        return out, attn_w


# ---------------------------
# 5) Loss：UncertaintyLoss + label smoothing（与你 notebook 对齐）
# ---------------------------
class UncertaintyLoss(nn.Module):
    def __init__(self, loss_types: List[str] = ["huber", "mse", "mse"]):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(len(loss_types)))
        self.loss_types = loss_types

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        task_losses = []

        for i in range(3):
            if self.loss_types[i] == "mse":
                task_loss = F.mse_loss(preds[:, i], targets[:, i], reduction="mean")
            elif self.loss_types[i] == "huber":
                task_loss = F.smooth_l1_loss(preds[:, i], targets[:, i], reduction="mean")
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_types[i]}")

            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * task_loss + self.log_vars[i]
            task_losses.append(task_loss.item())

        metrics = {
            "λ_loss": task_losses[0],
            "stokes_loss": task_losses[1],
            "QY_loss": task_losses[2],
            "regression_loss": float(sum(task_losses)),
            "reward": 0.0,
            "reward_factor": 0.0,
            "total_loss": float(total_loss.item()),
        }
        return total_loss, metrics


def apply_label_smoothing(
    y: torch.Tensor,
    lambda_eps: float = 0.01,
    stokes_eps: float = 0.05,
    qy_eps: float = 0.01,
    noise_type: str = "uniform",
) -> torch.Tensor:
    y_smooth = y.clone()
    if noise_type == "uniform":
        y_smooth[:, 0] += (torch.rand_like(y[:, 0]) * 2 - 1) * lambda_eps
        y_smooth[:, 1] += (torch.rand_like(y[:, 1]) * 2 - 1) * stokes_eps
        y_smooth[:, 2] += (torch.rand_like(y[:, 2]) * 2 - 1) * qy_eps
    elif noise_type == "normal":
        y_smooth[:, 0] += torch.randn_like(y[:, 0]) * lambda_eps
        y_smooth[:, 1] += torch.randn_like(y[:, 1]) * stokes_eps
        y_smooth[:, 2] += torch.randn_like(y[:, 2]) * qy_eps
    else:
        raise ValueError("Unsupported noise_type")

    y_smooth[:, 2] = torch.clamp(y_smooth[:, 2], 0, 1)
    return y_smooth


# ---------------------------
# 6) 训练/评估：日志、early stopping、scheduler
# ---------------------------
def _init_epoch_logs() -> Dict[str, float]:
    return {
        "λ_loss": 0.0,
        "stokes_loss": 0.0,
        "QY_loss": 0.0,
        "regression_loss": 0.0,
        "reward": 0.0,
        "total_loss": 0.0,
        "reward_factor": 0.0,
        "_n": 0,  # 样本数
    }


def _accumulate(logs: Dict[str, float], metrics: Dict[str, float], batch_size: int) -> None:
    for k in ["λ_loss", "stokes_loss", "QY_loss", "regression_loss", "reward", "total_loss", "reward_factor"]:
        logs[k] += float(metrics[k]) * batch_size
    logs["_n"] += batch_size


def _finalize(logs: Dict[str, float]) -> Dict[str, float]:
    n = max(int(logs["_n"]), 1)
    out = {k: logs[k] / n for k in ["λ_loss", "stokes_loss", "QY_loss", "regression_loss", "reward", "total_loss", "reward_factor"]}
    return out

TARGET_NAMES = ["Em_max_nm", "stoke_shift", "QY_%"]
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # compatible with older sklearn (no squared=)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = {}
    for j, name in enumerate(TARGET_NAMES):
        m[f"{name}_MAE"] = float(mean_absolute_error(y_true[:, j], y_pred[:, j]))
        m[f"{name}_RMSE"] = rmse(y_true[:, j], y_pred[:, j])
        m[f"{name}_R2"] = float(r2_score(y_true[:, j], y_pred[:, j]))
    m["MAE_mean"] = float(np.mean([m[f"{t}_MAE"] for t in TARGET_NAMES]))
    m["RMSE_mean"] = float(np.mean([m[f"{t}_RMSE"] for t in TARGET_NAMES]))
    m["R2_mean"] = float(np.mean([m[f"{t}_R2"] for t in TARGET_NAMES]))
    return m

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    logs = _init_epoch_logs()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred, _ = model(
                batch["graphs_1"], batch["graphs_2"],
                batch["fps_1"], batch["fps_2"],
                batch["desc_1"], batch["desc_2"],
                batch["global_feat"],
            )
            loss, metrics = criterion(pred, batch["y"])
            _accumulate(logs, metrics, batch["y"].size(0))
    return _finalize(logs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: bool = True,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    logs = _init_epoch_logs()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        pred, _ = model(
            batch["graphs_1"], batch["graphs_2"],
            batch["fps_1"], batch["fps_2"],
            batch["desc_1"], batch["desc_2"],
            batch["global_feat"],
        )

        y = apply_label_smoothing(batch["y"]) if label_smoothing else batch["y"]
        loss, metrics = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        _accumulate(logs, metrics, batch["y"].size(0))

    return _finalize(logs)


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 20  # early stopping on test regression_loss
    t0: int = 20        # CosineAnnealingWarmRestarts
    t_mult: int = 2
    eta_min: float = 5e-5
    label_smoothing: bool = True
    grad_clip: float = 1.0
    batch_size: int = 10


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig = TrainConfig(),
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    if criterion is None:
        criterion = UncertaintyLoss(loss_types=["huber", "mse", "mse"])

    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.t0, T_mult=cfg.t_mult, eta_min=cfg.eta_min
    )

    best_loss = float("inf")
    best_state = None
    bad = 0

    history = {"train": [], "test": []}

    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            label_smoothing=cfg.label_smoothing, grad_clip=cfg.grad_clip
        )
        test_metrics = evaluate(model, test_loader, criterion, device)

        scheduler.step(epoch)

        history["train"].append(train_metrics)
        history["test"].append(test_metrics)

        cur = test_metrics["regression_loss"]
        if cur < best_loss:
            best_loss = cur
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        # 打印（你也可以关掉）
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d} | "
            f"Train total={train_metrics['total_loss']:.4f} reg={train_metrics['regression_loss']:.4f} | "
            f"Test total={test_metrics['total_loss']:.4f} reg={test_metrics['regression_loss']:.4f} | "
            f"LR={lr_now:.6f}"
        )

        if bad >= cfg.patience:
            print(f"[Early Stopping] best test regression_loss={best_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_test_regression_loss": best_loss,
        "history": history,
    }


# ---------------------------
# 7) 一键构建：featurizer + loaders + model
# ---------------------------
def build_dual_gnn_pipeline(
    df_raw: pd.DataFrame,
    df_smiles: pd.DataFrame,
    solvent_polarity_data: Dict[str, Dict[str, float]],
    batch_size: int = 10,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[DualPrecursorGNN, DataLoader, DataLoader, CarbonDotFeaturizer]:
    df = data_process(df_raw)
    featurizer = CarbonDotFeaturizer(df_smiles=df_smiles, solvent_polarity_data=solvent_polarity_data)
    train_loader, test_loader = build_loaders(df, featurizer, batch_size=batch_size, test_size=test_size, seed=seed)
    model = DualPrecursorGNN()
    return model, train_loader, test_loader, featurizer

def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig = TrainConfig(),
    criterion: Optional[nn.Module] = None,
) -> Tuple[nn.Module, Dict[str, float]]:
    if criterion is None:
        criterion = UncertaintyLoss(loss_types=["huber", "mse", "mse"])

    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.t0, T_mult=cfg.t_mult, eta_min=cfg.eta_min
    )

    best_val = float("inf")
    best_state = None
    bad = 0

    for _epoch in range(cfg.epochs):
        # train
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(
                batch["graphs_1"], batch["graphs_2"],
                batch["fps_1"], batch["fps_2"],
                batch["desc_1"], batch["desc_2"],
                batch["global_feat"],
            )
            loss,metss = criterion(pred, batch["y"])
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # val
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(
                    batch["graphs_1"], batch["graphs_2"],
                    batch["fps_1"], batch["fps_2"],
                    batch["desc_1"], batch["desc_2"],
                    batch["global_feat"],
                )
                loss,metss = criterion(pred, batch["y"])
                losses.append(loss.item())
        val_loss = float(np.mean(losses))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval metrics
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch_cpu_y = batch["y"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(
                batch["graphs_1"], batch["graphs_2"],
                batch["fps_1"], batch["fps_2"],
                batch["desc_1"], batch["desc_2"],
                batch["global_feat"],
            ).detach().cpu().numpy()
            y_true.append(batch_cpu_y)
            y_pred.append(pred)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    metrics = {}
    # per-target MAE/RMSE/R2
    for i, name in enumerate(["Em_max_nm", "stoke_shift", "QY_%"]):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        metrics[f"{name}_MAE"] = float(mae)
        metrics[f"{name}_RMSE"] = rmse
        metrics[f"{name}_R2"] = float(r2)
    metrics["MAE_mean"] = float(np.mean([metrics["Em_max_nm_MAE"], metrics["stoke_shift_MAE"], metrics["QY_%_MAE"]]))
    metrics["RMSE_mean"] = float(np.mean([metrics["Em_max_nm_RMSE"], metrics["stoke_shift_RMSE"], metrics["QY_%_RMSE"]]))
    metrics["R2_mean"] = float(np.mean([metrics["Em_max_nm_R2"],metrics["stoke_shift_R2"],metrics["QY_%_R2"]]))
    metrics["val_loss"] = best_val
    return model, metrics

def cv5_compare(
    dataset,  # your torch Dataset returning the batch fields via collate_fn
    collate_fn,
    build_original: Callable[[], nn.Module],
    cfg= TrainConfig(),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    dataset: e.g., your CarbonDotDataset(...)
    collate_fn: your custom_collate_fn that returns dict with keys:
      graphs_1, graphs_2, fps_1, fps_2, desc_1, desc_2, global_feat, y
    """
    set_seed(seed)
    device = get_device()
    #loss_fn = MultiTaskMSE()
    loss_fn = UncertaintyLoss()
    idx = np.arange(len(dataset))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = {"original": []}
    for fold, (tr, va) in enumerate(kf.split(idx), 1):
        train_set = Subset(dataset, tr.tolist())
        val_set = Subset(dataset, va.tolist())

        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

        # Original
        m1 = build_original()
        _, met1 = train_one_fold(m1, train_loader, val_loader, device, cfg, loss_fn)
        results["original"].append(met1)

        print(f"\n[Fold {fold}/5] Original MAE_mean={met1['MAE_mean']:.4f} | R2_mean={met1['R2_mean']:.4f}")

    def summarize(lst: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        keys = lst[0].keys()
        out = {}
        for k in keys:
            arr = np.array([d[k] for d in lst], dtype=float)
            out[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}
        return out

    summary = {
        "original": summarize(results["original"])
    }
    return {"folds": results, "summary": summary}

