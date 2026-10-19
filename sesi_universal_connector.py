"""
sesi_universal_connector.py
=============================================================================

# Developer   : PAI and Yoon A Limsuwan — MSPS NETWORK
# ORCID       : 0009-0008-2374-0788
# GitHub      : yoonalimsuwan
# Contact     : msps4u@gmail.com
# License     : MIT
# Year        : 2026


SESI UNIVERSAL CONNECTOR — SPECIAL BRIDGE FILE
Implements Steps 2, 3, 4:
  Step 2: unit/self tests, finite checks, shape checks, gradient-friendly design
  Step 3: standardized input schema for all modules
  Step 4: metrics, fusion, and integration report

Connects:
  First 4 modules:
    - QMD
    - HCSN
    - SPTC
    - SESI Stealth Footprint Absorber

  Later 5 modules:
    - SESI Hypersonic Flight Control 6DOF
    - SESI EW Structural Contraction Filter
    - SESI Physics Core
    - Multi-Domain Countermeasure Engine
    - Omni-Spectral Drone vs Biological Discriminator

  Latest uploaded module:
    - Hypersonic Stealth Omni-Detector

Recommended rename for clean imports:
    sesi_physics_core.py
    sesi_stealth_footprint_absorber.py
    sesi_hypersonic_flight_control.py
    sesi_ew_structural_contraction_filter.py
    sesi_multi_domain_countermeasure_engine.py
    sesi_omni_spectral_drone_discriminator.py
    hypersonic_stealth_omni_detector.py
    qmd_hypothesis_tests.py
    satellite_free_hcsn.py
    sptc_hypothesis_tests.py

If you do not rename them, pass explicit paths:
    connector = UniversalSESIConnector(
        module_paths={
            "physics_core": "./sESI_PHYSIC_CORE_FIX.py",
            "stealth": "./sESI_SlALTH_FOOTPRINT_ ABSORBER_FIX (1).py",
            ...
        }
    )

License: MIT
=============================================================================
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "StandardSample",
    "UniversalSESIConnector",
    "make_synthetic_sample",
    "normalize_sample",
    "extract_pooled_features",
]


#=============================================================================
# 1. Module discovery / canonical names
#=============================================================================
CANONICAL_MODULE_NAMES: Dict[str, str] = {
    "physics_core": "sesi_physics_core",
    "stealth": "sesi_stealth_footprint_absorber",
    "flight": "sesi_hypersonic_flight_control",
    "ew": "sesi_ew_structural_contraction_filter",
    "countermeasure": "sesi_multi_domain_countermeasure_engine",
    "omni": "sesi_omni_spectral_drone_discriminator",
    "hypersonic_detector": "hypersonic_stealth_omni_detector",
    "hcsn": "satellite_free_hcsn",
    "qmd": "qmd_hypothesis_tests",
    "sptc": "sptc_hypothesis_tests",
}

MODULE_CANDIDATES: Dict[str, List[str]] = {
    "physics_core": [
        "sesi_physics_core",
        "sESI_PHYSIC_CORE_FIX",
    ],
    "stealth": [
        "sesi_stealth_footprint_absorber",
        "sESI_SlALTH_FOOTPRINT_ABSORBER_FIX",
    ],
    "flight": [
        "sesi_hypersonic_flight_control",
        "sESI_HYPERSONIC_FLIGHT_CONTROL_6DOF_RIGID_BODY_NEWTONIAN_AERO_FIX",
    ],
    "ew": [
        "sesi_ew_structural_contraction_filter",
        "sESI_EW_STRUCTURAL_CONTRACTION_FILTER_FIX",
    ],
    "countermeasure": [
        "sesi_multi_domain_countermeasure_engine",
        "mULTI_DOMAIN_COUNTERMEASURE_ENGINE_FIX",
    ],
    "omni": [
        "sesi_omni_spectral_drone_discriminator",
        "oMNI_SPECTRAL_DRONE_vsBIOLOGICAL_DISCRIMINATOR_FIX",
    ],
    "hypersonic_detector": [
        "hypersonic_stealth_omni_detector",
        "hYPERSONIC_STEALTH_OMNI_DETECTOR_FIX",
    ],
    "hcsn": [
        "satellite_free_hcsn",
        "satellite_Free_Hierarchical_Cloud_Superposition_Network_fixed",
    ],
    "qmd": [
        "qmd_hypothesis_tests",
        "qmd_hypothesis_tests_fixed",
    ],
    "sptc": [
        "sptc_hypothesis_tests",
        "sptc_hypothesis_tests_fixed",
    ],
}


def load_module_from_path(module_name: str, path: str):
    """
    Load a Python file as a module and register it in sys.modules under
    `module_name`, so that dependent imports such as
    `from sesi_physics_core import ...` can resolve correctly.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


#=============================================================================
# 2. Standardized sample schema — Step 3
#=============================================================================
@dataclass
class StandardSample:
    """
    Standard data contract for all modules.

    5D fields use shape:
        velocity_field : [B, 3, D, H, W]
        thermal_field  : [B, 1, D, H, W]
        pressure_field : [B, 1, D, H, W]
        density_field  : [B, 1, D, H, W]
        em_field       : [B, 3, D, H, W]
        emf_coupling   : [B, 1, D, H, W]
        sigma_skin     : [B, 1, D, H, W]

    Flight:
        flight_state   : [B, 12]
        delta_e_cmd    : [B]
        thrust         : [B]

    EW:
        ew_raw         : [B, F]
        ew_jam         : [B, F]

    HCSN:
        hcsn_signal    : [B, N, d_in]
        hcsn_target    : [B, N, d_out]

    QMD:
        qmd_D          : [B, T, d_D]
        qmd_E          : [B, T, d_E]

    SPTC:
        sptc_substrate : [B, d_substrate]

    Countermeasure:
        target_position        : [B, 3]
        target_velocity        : [B, 3]
        target_mass            : [B]
        predicted_trajectory   : [B, H, 3]
        gravimeter_weights     : [K]
        muon_sensor_grid       : [K]
        dew_focal_controls     : [B, 3]
    """

    # Field physics
    velocity_field: torch.Tensor = None
    thermal_field: torch.Tensor = None
    pressure_field: torch.Tensor = None
    density_field: torch.Tensor = None
    em_field: torch.Tensor = None
    emf_coupling: torch.Tensor = None
    sigma_skin: torch.Tensor = None

    # Flight
    flight_state: torch.Tensor = None
    delta_e_cmd: torch.Tensor = None
    thrust: torch.Tensor = None

    # EW
    ew_raw: torch.Tensor = None
    ew_jam: torch.Tensor = None

    # HCSN
    hcsn_signal: torch.Tensor = None
    hcsn_target: torch.Tensor = None

    # QMD
    qmd_D: torch.Tensor = None
    qmd_E: torch.Tensor = None

    # SPTC
    sptc_substrate: torch.Tensor = None

    # Countermeasure
    target_position: torch.Tensor = None
    target_velocity: torch.Tensor = None
    target_mass: torch.Tensor = None
    predicted_trajectory: torch.Tensor = None
    gravimeter_weights: torch.Tensor = None
    muon_sensor_grid: torch.Tensor = None
    dew_focal_controls: torch.Tensor = None

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "StandardSample":
        updates = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                if v.is_floating_point() and dtype is not None:
                    updates[k] = v.to(device=device, dtype=dtype)
                else:
                    updates[k] = v.to(device=device)
            else:
                updates[k] = v
        return replace(self, **updates)

    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        out = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                out[k] = tuple(v.shape)
        return out

    def validate(self) -> Dict[str, Any]:
        errors: List[str] = []
        batch_size = None

        def set_batch(t: torch.Tensor, name: str):
            nonlocal batch_size
            if t is None:
                return
            if batch_size is None:
                batch_size = t.shape[0]
            elif t.shape[0] != batch_size:
                errors.append(f"{name} batch {t.shape[0]} != expected {batch_size}")

        # Core 5D fields
        core_5d = {
            "velocity_field": self.velocity_field,
            "thermal_field": self.thermal_field,
            "pressure_field": self.pressure_field,
            "density_field": self.density_field,
            "em_field": self.em_field,
            "emf_coupling": self.emf_coupling,
            "sigma_skin": self.sigma_skin,
        }

        spatial_shape = None
        for name, t in core_5d.items():
            if t is None:
                errors.append(f"{name} is None")
                continue
            set_batch(t, name)
            if t.dim() != 5:
                errors.append(f"{name} must be 5D [B,C,D,H,W], got {tuple(t.shape)}")
                continue
            if spatial_shape is None:
                spatial_shape = tuple(t.shape[2:])
            elif tuple(t.shape[2:]) != spatial_shape:
                errors.append(f"{name} spatial shape {tuple(t.shape[2:])} != {spatial_shape}")

        if self.velocity_field is not None and self.velocity_field.shape[1] != 3:
            errors.append("velocity_field must have 3 channels")
        for name in ["thermal_field", "pressure_field", "density_field", "emf_coupling", "sigma_skin"]:
            t = getattr(self, name)
            if t is not None and t.shape[1] != 1:
                errors.append(f"{name} must have 1 channel")
        if self.em_field is not None and self.em_field.shape[1] != 3:
            errors.append("em_field must have 3 channels")

        # Flight
        if self.flight_state is not None:
            set_batch(self.flight_state, "flight_state")
            if self.flight_state.dim() != 2 or self.flight_state.shape[-1] != 12:
                errors.append("flight_state must be [B,12]")
        if self.delta_e_cmd is not None:
            set_batch(self.delta_e_cmd, "delta_e_cmd")
            if self.delta_e_cmd.dim() != 1:
                errors.append("delta_e_cmd must be [B]")
        if self.thrust is not None:
            set_batch(self.thrust, "thrust")
            if self.thrust.dim() != 1:
                errors.append("thrust must be [B]")

        # EW
        if self.ew_raw is not None:
            set_batch(self.ew_raw, "ew_raw")
            if self.ew_raw.dim() != 2:
                errors.append("ew_raw must be [B,F]")
        if self.ew_jam is not None:
            set_batch(self.ew_jam, "ew_jam")
            if self.ew_jam.dim() != 2:
                errors.append("ew_jam must be [B,F]")
        if self.ew_raw is not None and self.ew_jam is not None:
            if self.ew_raw.shape != self.ew_jam.shape:
                errors.append("ew_raw and ew_jam must have same shape")

        # HCSN
        if self.hcsn_signal is not None:
            set_batch(self.hcsn_signal, "hcsn_signal")
            if self.hcsn_signal.dim() != 3:
                errors.append("hcsn_signal must be [B,N,d_in]")
        if self.hcsn_target is not None:
            set_batch(self.hcsn_target, "hcsn_target")
            if self.hcsn_target.dim() != 3:
                errors.append("hcsn_target must be [B,N,d_out]")
        if self.hcsn_signal is not None and self.hcsn_target is not None:
            if self.hcsn_signal.shape[1] != self.hcsn_target.shape[1]:
                errors.append("hcsn_signal and hcsn_target sequence lengths must match")

        # QMD
        if self.qmd_D is not None:
            set_batch(self.qmd_D, "qmd_D")
            if self.qmd_D.dim() != 3:
                errors.append("qmd_D must be [B,T,d_D]")
        if self.qmd_E is not None:
            set_batch(self.qmd_E, "qmd_E")
            if self.qmd_E.dim() != 3:
                errors.append("qmd_E must be [B,T,d_E]")
        if self.qmd_D is not None and self.qmd_E is not None:
            if self.qmd_D.shape[1] != self.qmd_E.shape[1]:
                errors.append("qmd_D and qmd_E sequence lengths must match")

        # SPTC
        if self.sptc_substrate is not None:
            set_batch(self.sptc_substrate, "sptc_substrate")
            if self.sptc_substrate.dim() != 2:
                errors.append("sptc_substrate must be [B,d_substrate]")

        # Countermeasure
        if self.target_position is not None:
            set_batch(self.target_position, "target_position")
            if self.target_position.dim() != 2 or self.target_position.shape[-1] != 3:
                errors.append("target_position must be [B,3]")
        if self.target_velocity is not None:
            set_batch(self.target_velocity, "target_velocity")
            if self.target_velocity.dim() != 2 or self.target_velocity.shape[-1] != 3:
                errors.append("target_velocity must be [B,3]")
        if self.target_mass is not None:
            set_batch(self.target_mass, "target_mass")
            if self.target_mass.dim() != 1:
                errors.append("target_mass must be [B]")
        if self.predicted_trajectory is not None:
            set_batch(self.predicted_trajectory, "predicted_trajectory")
            if self.predicted_trajectory.dim() != 3 or self.predicted_trajectory.shape[-1] != 3:
                errors.append("predicted_trajectory must be [B,H,3]")
        if self.dew_focal_controls is not None:
            set_batch(self.dew_focal_controls, "dew_focal_controls")
            if self.dew_focal_controls.dim() != 2 or self.dew_focal_controls.shape[-1] != 3:
                errors.append("dew_focal_controls must be [B,3]")

        return {
            "ok": len(errors) == 0,
            "batch_size": batch_size,
            "errors": errors,
        }


def make_synthetic_sample(
    batch_size: int = 2,
    grid: int = 8,
    ew_dim: int = 64,
    hcsn_len: int = 16,
    hcsn_d_in: int = 16,
    hcsn_d_out: int = 16,
    qmd_len: int = 8,
    qmd_d_D: int = 8,
    qmd_d_E: int = 4,
    sptc_dim: int = 12,
    horizon: int = 12,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> StandardSample:
    """
    Create one standardized synthetic sample that covers all module inputs.
    Replace this with real data loaders later.
    """
    device = torch.device(device)
    B, D, H, W = batch_size, grid, grid, grid

    velocity = torch.randn(B, 3, D, H, W, device=device, dtype=dtype) * 200.0
    thermal = torch.rand(B, 1, D, H, W, device=device, dtype=dtype) * 30.0 + 240.0
    pressure = torch.rand(B, 1, D, H, W, device=device, dtype=dtype) * 1200.0 + 100.0
    density = torch.rand(B, 1, D, H, W, device=device, dtype=dtype) * 0.15 + 0.01
    em = torch.randn(B, 3, D, H, W, device=device, dtype=dtype)
    emf = torch.randn(B, 1, D, H, W, device=device, dtype=dtype) * 0.05
    sigma = torch.ones(B, 1, D, H, W, device=device, dtype=dtype)

    flight_state = torch.zeros(B, 12, device=device, dtype=dtype)
    flight_state[:, 2] = 30000.0
    flight_state[:, 3] = 2000.0
    delta_e_cmd = torch.zeros(B, device=device, dtype=dtype)
    thrust = torch.full((B,), 5000.0, device=device, dtype=dtype)

    ew_raw = torch.randn(B, ew_dim, device=device, dtype=dtype)
    ew_jam = torch.randn(B, ew_dim, device=device, dtype=dtype) * 0.4

    hcsn_signal = torch.randn(B, hcsn_len, hcsn_d_in, device=device, dtype=dtype)
    hcsn_target = torch.tanh(hcsn_signal @ torch.randn(hcsn_d_in, hcsn_d_out, device=device, dtype=dtype))

    qmd_D = torch.randn(B, qmd_len, qmd_d_D, device=device, dtype=dtype)
    qmd_E = torch.randn(B, qmd_len, qmd_d_E, device=device, dtype=dtype)

    sptc_substrate = torch.randn(B, sptc_dim, device=device, dtype=dtype)

    target_position = torch.tensor(
        [[1.0e5, 2.0e5, 3.0e4],
         [1.2e5, 2.2e5, 2.8e4]],
        device=device,
        dtype=dtype,
    )[:B]
    target_velocity = torch.tensor(
        [[2000.0, 500.0, -100.0],
         [1800.0, 400.0, -80.0]],
        device=device,
        dtype=dtype,
    )[:B]
    target_mass = torch.tensor([1500.0, 1800.0], device=device, dtype=dtype)[:B]

    predicted_trajectory = (
        target_position.unsqueeze(1).expand(B, horizon, 3).clone()
        + torch.randn(B, horizon, 3, device=device, dtype=dtype) * 100.0
    )

    gravimeter_weights = torch.randn(4, device=device, dtype=dtype)
    muon_sensor_grid = torch.randn(4, device=device, dtype=dtype)
    dew_focal_controls = target_position + 100.0

    return StandardSample(
        velocity_field=velocity,
        thermal_field=thermal,
        pressure_field=pressure,
        density_field=density,
        em_field=em,
        emf_coupling=emf,
        sigma_skin=sigma,
        flight_state=flight_state,
        delta_e_cmd=delta_e_cmd,
        thrust=thrust,
        ew_raw=ew_raw,
        ew_jam=ew_jam,
        hcsn_signal=hcsn_signal,
        hcsn_target=hcsn_target,
        qmd_D=qmd_D,
        qmd_E=qmd_E,
        sptc_substrate=sptc_substrate,
        target_position=target_position,
        target_velocity=target_velocity,
        target_mass=target_mass,
        predicted_trajectory=predicted_trajectory,
        gravimeter_weights=gravimeter_weights,
        muon_sensor_grid=muon_sensor_grid,
        dew_focal_controls=dew_focal_controls,
    )


def normalize_sample(sample: StandardSample) -> StandardSample:
    """
    Step 3: sanitize and standardize tensor ranges before passing to modules.
    This is intentionally conservative: it does not destroy physical units,
    but it removes NaN/Inf and enforces positive fields where needed.
    """
    updates = {}

    def clean(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        if not t.is_floating_point():
            return t
        return torch.nan_to_num(t, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

    for k, v in sample.__dict__.items():
        if torch.is_tensor(v):
            updates[k] = clean(v)

    # Positive physical fields
    if updates.get("thermal_field") is not None:
        updates["thermal_field"] = updates["thermal_field"].clamp_min(1.0)
    if updates.get("pressure_field") is not None:
        updates["pressure_field"] = updates["pressure_field"].clamp_min(1.0)
    if updates.get("density_field") is not None:
        updates["density_field"] = updates["density_field"].clamp_min(1.0e-6)
    if updates.get("sigma_skin") is not None:
        updates["sigma_skin"] = updates["sigma_skin"].clamp_min(1.0e-6)
    if updates.get("target_mass") is not None:
        updates["target_mass"] = updates["target_mass"].clamp_min(1.0e-3)

    return replace(sample, **updates)


def extract_pooled_features(sample: StandardSample) -> torch.Tensor:
    """
    Step 3/4: extract a compact pooled feature vector from the standard sample.
    Useful for QMD evidence, SPTC substrate, or global decision features.
    """
    feats: List[torch.Tensor] = []

    def pool_5d(t: Optional[torch.Tensor]) -> torch.Tensor:
        if t is None:
            return None
        return t.pow(2).mean(dim=tuple(range(1, t.dim()))).sqrt()

    for name in [
        "velocity_field",
        "thermal_field",
        "pressure_field",
        "density_field",
        "em_field",
        "emf_coupling",
        "sigma_skin",
    ]:
        t = getattr(sample, name)
        if t is not None:
            feats.append(pool_5d(t))

    if sample.flight_state is not None:
        feats.append(sample.flight_state.norm(dim=-1, keepdim=False))

    if sample.ew_raw is not None:
        feats.append(sample.ew_raw.pow(2).mean(dim=-1, keepdim=False))
    if sample.ew_jam is not None:
        feats.append(sample.ew_jam.pow(2).mean(dim=-1, keepdim=False))

    if sample.hcsn_signal is not None:
        feats.append(sample.hcsn_signal.pow(2).mean(dim=(1, 2), keepdim=False))

    if sample.qmd_D is not None:
        feats.append(sample.qmd_D.pow(2).mean(dim=(1, 2), keepdim=False))

    if sample.sptc_substrate is not None:
        feats.append(sample.sptc_substrate.norm(dim=-1, keepdim=False))

    if not feats:
        raise ValueError("No tensors available to extract pooled features.")

    return torch.stack(feats, dim=-1)


#=============================================================================
# 3. Metrics — Step 4
#=============================================================================
def binary_metrics(pred: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Step 4: simple binary decision metrics.
    If target is None, returns confidence/positive-rate only.
    """
    if pred is None:
        return {"available": False}

    p = pred.detach().float().reshape(-1)
    p_prob = torch.sigmoid(p) if p.abs().max() > 1.5 else p
    p_bin = (p_prob > 0.5).float()

    out = {
        "available": True,
        "mean_prob": float(p_prob.mean().item()),
        "std_prob": float(p_prob.std(unbiased=False).item()),
        "positive_rate": float(p_bin.mean().item()),
    }

    if target is not None:
        y = target.detach().float().reshape(-1)
        tp = ((p_bin == 1) & (y == 1)).sum().item()
        fp = ((p_bin == 1) & (y == 0)).sum().item()
        tn = ((p_bin == 0) & (y == 0)).sum().item()
        fn = ((p_bin == 0) & (y == 1)).sum().item()
        eps = 1e-12

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(eps, precision + recall)
        acc = (tp + tn) / max(1, tp + fp + tn + fn)

        out.update(
            {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    return out


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Step 4: simple regression metrics for HCSN or continuous outputs."""
    if pred is None or target is None:
        return {"available": False}

    p = pred.detach().float()
    t = target.detach().float()
    mae = F.l1_loss(p, t).item()
    mse = F.mse_loss(p, t).item()
    denom = t.pow(2).mean().clamp_min(1e-12)
    nrmse = math.sqrt(mse) / denom.sqrt().item()

    return {
        "available": True,
        "mae": float(mae),
        "mse": float(mse),
        "nrmse": float(nrmse),
    }


def check_finite(obj: Any, prefix: str = "") -> List[str]:
    """Step 2: recursively find non-finite tensors."""
    problems: List[str] = []

    if torch.is_tensor(obj):
        if not torch.isfinite(obj).all():
            problems.append(prefix or "tensor")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            problems.extend(check_finite(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            problems.extend(check_finite(v, f"{prefix}[{i}]"))

    return problems


def tensor_stats(t: Optional[torch.Tensor]) -> Dict[str, float]:
    if t is None or not torch.is_tensor(t):
        return {"available": False}
    x = t.detach().float()
    if x.numel() == 0:
        return {"available": False}
    return {
        "available": True,
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


#=============================================================================
# 4. Universal connector — Step 2 + Step 3 + Step 4 bridge
#=============================================================================
class UniversalSESIConnector(nn.Module):
    """
    Special bridge module that connects all SESI modules using one
    standardized input sample.

    This module does not replace the original modules. It dynamically imports
    them, adapts their inputs, runs them, checks outputs, and produces a
    unified report.
    """

    def __init__(
        self,
        module_paths: Optional[Dict[str, str]] = None,
        enable: Optional[Dict[str, bool]] = None,
        hcsn_d_in: int = 16,
        hcsn_d_out: int = 16,
        countermeasure_grid: Tuple[int, int, int] = (8, 8, 8),
        use_detached_bridge: bool = False,
        build: bool = True,
    ) -> None:
        super().__init__()
        self.module_paths = module_paths or {}
        self.hcsn_d_in = hcsn_d_in
        self.hcsn_d_out = hcsn_d_out
        self.countermeasure_grid = countermeasure_grid
        self.use_detached_bridge = use_detached_bridge

        self.enable = {
            "physics_core": True,
            "stealth": True,
            "flight": True,
            "ew": True,
            "countermeasure": True,
            "omni": True,
            "hypersonic_detector": True,
            "hcsn": True,
            "qmd": True,
            "sptc": True,
        }
        if enable is not None:
            self.enable.update(enable)

        self._libs: Dict[str, Any] = {}
        self._classes: Dict[str, Any] = {}
        self._import_errors: Dict[str, str] = {}
        self._modules = nn.ModuleDict()

        if build:
            self.build_modules()

    #=========================================================================
    # Dynamic import helpers
    #=========================================================================
    def _load_module(self, key: str):
        if key in self._libs:
            return self._libs[key]

        if not self.enable.get(key, False):
            self._import_errors[key] = "disabled"
            return None

        canonical = CANONICAL_MODULE_NAMES[key]
        path = self.module_paths.get(key, None)

        try:
            if path is not None:
                mod = load_module_from_path(canonical, path)
            else:
                mod = None
                candidates = [canonical] + MODULE_CANDIDATES.get(key, [])
                last_error = None
                for cand in candidates:
                    try:
                        mod = importlib.import_module(cand)
                        break
                    except Exception as e:
                        last_error = e
                        continue
                if mod is None:
                    raise ImportError(f"Could not import {key}: {last_error}")

            # Register canonical alias for dependent imports.
            sys.modules.setdefault(canonical, mod)
            self._libs[key] = mod
            return mod

        except Exception as e:
            self._import_errors[key] = str(e)
            return None

    #=========================================================================
    # Build all modules
    #=========================================================================
    def build_modules(self) -> None:
        self._modules.clear()
        self._libs.clear()
        self._classes.clear()
        self._import_errors.clear()

        # Physics core is a library, not nn.Module.
        self._load_module("physics_core")

        # Stealth absorber
        if self.enable["stealth"]:
            mod = self._load_module("stealth")
            if mod is not None:
                cls = getattr(mod, "SESIStealthFootprintAbsorberV3", None)
                if cls is not None:
                    self._modules["stealth"] = cls()

        # Flight control
        if self.enable["flight"]:
            mod = self._load_module("flight")
            if mod is not None:
                cls = getattr(mod, "SESIHypersonicFlightController6DOF", None)
                if cls is not None:
                    self._modules["flight"] = cls()

        # EW filter
        if self.enable["ew"]:
            mod = self._load_module("ew")
            if mod is not None:
                cls = getattr(mod, "EWStructuralContractionModuleV3", None)
                if cls is not None:
                    self._modules["ew"] = cls()

        # Omni discriminator
        if self.enable["omni"]:
            mod = self._load_module("omni")
            if mod is not None:
                cls = getattr(mod, "OmniTargetDetectorV3", None)
                if cls is not None:
                    self._modules["omni"] = cls()

        # Latest hypersonic stealth omni detector
        if self.enable["hypersonic_detector"]:
            mod = self._load_module("hypersonic_detector")
            if mod is not None:
                cls = getattr(mod, "HypersonicStealthOmniDetectorPhysics", None)
                if cls is not None:
                    self._modules["hypersonic_detector"] = cls()

        # HCSN
        if self.enable["hcsn"]:
            mod = self._load_module("hcsn")
            if mod is not None:
                cls = getattr(mod, "SatelliteFreeHCSN", None)
                if cls is not None:
                    self._modules["hcsn"] = cls(
                        d_in=self.hcsn_d_in,
                        d_out=self.hcsn_d_out,
                        d_model=64,
                        d_meta=32,
                        d_base=32,
                        n_meta_branches=4,
                        n_measurement_outcomes=4,
                        n_scales=4,
                        n_fields=6,
                        n_heads=4,
                        depth=2,
                        dropout=0.1,
                    )

        # QMD
        if self.enable["qmd"]:
            mod = self._load_module("qmd")
            if mod is not None:
                cfg_cls = getattr(mod, "QMDConfig", None)
                model_cls = getattr(mod, "QMDModel", None)
                if cfg_cls is not None and model_cls is not None:
                    cfg = cfg_cls()
                    cfg.enable_quantum = False
                    self._modules["qmd"] = model_cls(cfg)

        # SPTC
        if self.enable["sptc"]:
            mod = self._load_module("sptc")
            if mod is not None:
                cfg_cls = getattr(mod, "SPTCConfig", None)
                model_cls = getattr(mod, "SPTCModel", None)
                if cfg_cls is not None and model_cls is not None:
                    cfg = cfg_cls()
                    self._modules["sptc"] = model_cls(cfg)

        # Countermeasure engine
        if self.enable["countermeasure"]:
            mod = self._load_module("countermeasure")
            if mod is not None:
                cls = getattr(mod, "DifferentiableCountermeasureEngineV2", None)
                target_cls = getattr(mod, "TargetState", None)
                params_cls = getattr(mod, "DefenseParameters", None)
                if cls is not None:
                    self._modules["countermeasure"] = cls(grid_resolution=self.countermeasure_grid)
                if target_cls is not None:
                    self._classes["TargetState"] = target_cls
                if params_cls is not None:
                    self._classes["DefenseParameters"] = params_cls

    #=========================================================================
    # Safe forward utilities
    #=========================================================================
    def _safe_call(self, fn, *args, **kwargs) -> Tuple[Any, Optional[str]]:
        try:
            return fn(*args, **kwargs), None
        except Exception as e:
            return None, str(e)

    def forward_stealth(self, sample: StandardSample) -> Dict[str, Any]:
        model = self._modules.get("stealth", None)
        if model is None:
            return {"available": False, "error": self._import_errors.get("stealth", "not built")}
        out, err = self._safe_call(model, sample.velocity_field, sample.sigma_skin, return_metrics=True)
        if err is not None:
            return {"available": False, "error": err}
        out = dict(out)
        out["available"] = True
        return out

    def forward_flight(self, sample: StandardSample) -> Dict[str, Any]:
        model = self._modules.get("flight", None)
        if model is None:
            return {"available": False, "error": self._import_errors.get("flight", "not built")}
        out, err = self._safe_call(
            model,
            sample.flight_state,
            sample.delta_e_cmd,
            thrust=sample.thrust,
            dt=1.0e-3,
            substeps=2,
            return_metrics=True,
        )
        if err is not None:
            return {"available": False, "error": err}
        out = dict(out)
        out["available"] = True
        return out

    def forward_ew(self, sample: StandardSample) -> Dict[str, Any]:
        model = self._modules.get("ew", None)
        if model is None:
            return {"available": False, "error": self._import_errors.get("ew", "not built")}
        out, err = self._safe_call(
            model,
            sample.ew_raw,
            sample.ew_jam,
            dt=1.0e-6,
            return_metrics=True,
        )
        if err is not None:
            return {"available": False, "error": err}
        out = dict(out)
        out["available"] = True
        return out

    def forward_omni(self, sample: StandardSample) -> Dict[str, Any]:
        model = self._modules.get("omni", None)
        if model is None:
            return {"available": False, "error": self._import_errors.get("omni", "not built")}
        out, err = self._safe_call(
            model,
            sample.velocity_field,
            sample.thermal_field,
            sample.em_field,
            sample.density_field,
            sample.emf_coupling,
            dt=1.0e-3,
            return_metrics=True,
        )
        if err is not None:
            return {"available": False, "error": err}
        out = dict(out)
        out["available"] = True
        return out

    def forward_hypersonic_detector(self, sample: StandardSample) -> Dict[str, Any]:
        model = self._modules.get("hypersonic_detector", None)
        if model is None:
            return {"available": False, "error": self._import_errors.get("hypersonic_detector", "not built")}
        out, err = self._safe_call(
            model,
            sample.velocity_field,
            sample.thermal_field,
            sample.pressure_field,
            sample.density_field,
            sample.em_field,
            sample.emf_coupling,
            range_m=None,
            dt=1.0e-3,
            return_metrics=True,
        )
        if err is not None:
            return {"available": False, "error": err}
        out = dict(out)
        out["available"] = True
        return out

    def forward_countermeasure(self, sample: StandardSample) -> Dict[str, Any]:
        model = self._modules.get("countermeasure", None)
        target_cls = self._classes.get("TargetState", None)
        params_cls = self._classes.get("DefenseParameters", None)

        if model is None or target_cls is None or params_cls is None:
            return {"available": False, "error": self._import_errors.get("countermeasure", "not built")}

        target = target_cls(
            position=sample.target_position,
            velocity=sample.target_velocity,
            mass=sample.target_mass,
        )
        params = params_cls(
            gravimeter_weights=sample.gravimeter_weights,
            muon_sensor_grid=sample.muon_sensor_grid,
            dew_focal_controls=sample.dew_focal_controls,
        )

        out, err = self._safe_call(model, target, params, sample.predicted_trajectory)
        if err is not None:
            return {"available": False, "error": err}

        total_loss, metrics = out
        result = dict(metrics)
        result["available"] = True
        result["total_loss"] = total_loss
        return result

    def forward_hcsn(self, sample: StandardSample) -> Dict[str, Any]:
        model = self._modules.get("hcsn", None)
        if model is None:
            return {"available": False, "error": self._import_errors.get("hcsn", "not built")}

        out, err = self._safe_call(model, sample.hcsn_signal, return_aux=False)
        if err is not None:
            return {"available": False, "error": err}

        loss = F.mse_loss(out, sample.hcsn_target)
        return {
            "available": True,
            "output": out,
            "mse_loss": loss,
        }

    def forward_qmd(self, sample: StandardSample) -> Dict[str, Any]:
        model = self._modules.get("qmd", None)
        if model is None:
            return {"available": False, "error": self._import_errors.get("qmd", "not built")}

        out, err = self._safe_call(model, sample.qmd_D, sample.qmd_E, sample=False)
        if err is not None:
            return {"available": False, "error": err}

        out = dict(out)
        out["available"] = True
        return out

    def forward_sptc(self, sample: StandardSample) -> Dict[str, Any]:
        model = self._modules.get("sptc", None)
        if model is None:
            return {"available": False, "error": self._import_errors.get("sptc", "not built")}

        out, err = self._safe_call(model, sample.sptc_substrate, tau=0.7)
        if err is not None:
            return {"available": False, "error": err}

        out = dict(out)
        out["available"] = True
        return out

    #=========================================================================
    # Step 4: fusion
    #=========================================================================
    def _batch_vector(self, x: Optional[torch.Tensor], B: int) -> Optional[torch.Tensor]:
        if x is None or not torch.is_tensor(x):
            return None
        return x.detach().float().reshape(B, -1).mean(dim=1)

    def fuse_decisions(self, sample: StandardSample, reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        B = sample.velocity_field.shape[0]
        vecs: List[torch.Tensor] = []
        names: List[str] = []

        candidates = [
            ("hypersonic_detector", "is_hypersonic_drone"),
            ("omni", "is_drone"),
            ("ew", "is_true_target"),
            ("stealth", "stealth_target_achieved"),
        ]

        for module_name, key in candidates:
            rep = reports.get(module_name, {})
            if rep.get("available", False) and key in rep:
                v = self._batch_vector(rep[key], B)
                if v is not None:
                    vecs.append(v)
                    names.append(module_name)

        if not vecs:
            return {"available": False, "source_modules": []}

        stacked = torch.stack(vecs, dim=1)  # [B, n_modules]
        fused = stacked.mean(dim=1)         # [B]

        return {
            "available": True,
            "source_modules": names,
            "fused_decision": fused,
            "binary_metrics": binary_metrics(fused),
            "per_module_mean": {
                name: float(vec.mean().item()) for name, vec in zip(names, vecs)
            },
        }

    #=========================================================================
    # Main bridge forward
    #=========================================================================
    def forward(self, sample: StandardSample) -> Dict[str, Any]:
        """
        Full bridge pipeline:
          1. normalize sample
          2. run stealth absorber
          3. bridge absorbed velocity into sensor modules
          4. run latest hypersonic detector + omni discriminator + EW + flight
          5. run countermeasure / HCSN / QMD / SPTC
          6. fuse decisions and return report
        """
        sample = normalize_sample(sample)
        validation = sample.validate()

        report: Dict[str, Any] = {
            "validation": validation,
            "shapes": sample.shapes(),
            "modules": {},
            "finite_problems": [],
        }

        # 1) Stealth first: it can modify the velocity signature.
        stealth_out = self.forward_stealth(sample)
        report["modules"]["stealth"] = stealth_out

        if stealth_out.get("available", False) and "absorbed_velocity_field" in stealth_out:
            bridged_velocity = stealth_out["absorbed_velocity_field"]
            if self.use_detached_bridge:
                bridged_velocity = bridged_velocity.detach()
        else:
            bridged_velocity = sample.velocity_field

        bridged_sample = replace(sample, velocity_field=bridged_velocity)

        # 2) Sensor / discrimination modules
        report["modules"]["hypersonic_detector"] = self.forward_hypersonic_detector(bridged_sample)
        report["modules"]["omni"] = self.forward_omni(bridged_sample)
        report["modules"]["ew"] = self.forward_ew(bridged_sample)
        report["modules"]["flight"] = self.forward_flight(bridged_sample)

        # 3) Higher-level / physics / cognition / communication modules
        report["modules"]["countermeasure"] = self.forward_countermeasure(bridged_sample)
        report["modules"]["hcsn"] = self.forward_hcsn(bridged_sample)
        report["modules"]["qmd"] = self.forward_qmd(bridged_sample)
        report["modules"]["sptc"] = self.forward_sptc(bridged_sample)

        # 4) Step 2: finite checks on module outputs
        for name, out in report["modules"].items():
            problems = check_finite(out, prefix=name)
            report["finite_problems"].extend(problems)
            out["finite_ok"] = len(problems) == 0

        # 5) Step 4: fusion and metrics
        report["fusion"] = self.fuse_decisions(bridged_sample, report["modules"])

        if report["modules"].get("hcsn", {}).get("available", False):
            report["metrics_hcsn"] = regression_metrics(
                report["modules"]["hcsn"]["output"],
                bridged_sample.hcsn_target,
            )

        report["pooled_features_shape"] = tuple(extract_pooled_features(bridged_sample).shape)
        report["import_errors"] = dict(self._import_errors)

        return report

    #=========================================================================
    # Step 2: self-test / unit-test suite
    #=========================================================================
    def run_self_test(self, batch_size: int = 2, device: str = "cpu") -> Dict[str, Any]:
        torch.manual_seed(0)
        self.train()

        sample = make_synthetic_sample(batch_size=batch_size, device=device)
        report: Dict[str, Any] = {
            "step2_unit_tests": {},
            "step3_standardization": {},
            "step4_metrics": {},
        }

        # Step 3: schema validation
        validation = sample.validate()
        report["step3_standardization"]["raw_validation"] = validation
        report["step3_standardization"]["raw_shapes"] = sample.shapes()

        sample = normalize_sample(sample)
        validation_norm = sample.validate()
        report["step3_standardization"]["normalized_validation"] = validation_norm
        report["step3_standardization"]["normalized_shapes"] = sample.shapes()

        pooled = extract_pooled_features(sample)
        report["step3_standardization"]["pooled_features"] = tensor_stats(pooled)

        # Step 2: physics core differentiability test if available
        physics = self._load_module("physics_core")
        if physics is not None:
            try:
                soft_floor = getattr(physics, "soft_floor", None)
                if soft_floor is not None:
                    x = torch.tensor([0.0, 0.2, 1.0], requires_grad=True)
                    y = soft_floor(x, 0.1).sum()
                    y.backward()
                    grad_ok = x.grad is not None and torch.isfinite(x.grad).all().item()
                    report["step2_unit_tests"]["physics_core_soft_floor"] = {
                        "pass": bool(grad_ok),
                    }

                radar_range_equation = getattr(physics, "radar_range_equation", None)
                if radar_range_equation is not None:
                    R1 = torch.tensor(1.0e3)
                    R2 = torch.tensor(1.0e4)
                    P1 = radar_range_equation(
                        P_t=torch.tensor(1.0e4),
                        G=torch.tensor(100.0),
                        sigma_rcs=torch.tensor(0.1),
                        lam=torch.tensor(0.03),
                        R=R1,
                    )
                    P2 = radar_range_equation(
                        P_t=torch.tensor(1.0e4),
                        G=torch.tensor(100.0),
                        sigma_rcs=torch.tensor(0.1),
                        lam=torch.tensor(0.03),
                        R=R2,
                    )
                    report["step2_unit_tests"]["physics_core_radar_monotonic"] = {
                        "pass": bool(P1.item() > P2.item()),
                        "P_near": float(P1.item()),
                        "P_far": float(P2.item()),
                    }
            except Exception as e:
                report["step2_unit_tests"]["physics_core"] = {"pass": False, "error": str(e)}
        else:
            report["step2_unit_tests"]["physics_core"] = {
                "pass": False,
                "error": self._import_errors.get("physics_core", "not available"),
            }

        # Full bridge pipeline
        try:
            bridge_report = self.forward(sample)
            report["bridge"] = bridge_report
            report["step2_unit_tests"]["bridge_forward"] = {
                "pass": len(bridge_report.get("finite_problems", [])) == 0,
                "finite_problems": bridge_report.get("finite_problems", []),
            }

            # Step 4: global decision metrics
            fusion = bridge_report.get("fusion", {})
            if fusion.get("available", False):
                report["step4_metrics"]["fusion"] = fusion.get("binary_metrics", {})
            else:
                report["step4_metrics"]["fusion"] = {"available": False}

            if bridge_report.get("metrics_hcsn", None) is not None:
                report["step4_metrics"]["hcsn"] = bridge_report["metrics_hcsn"]

        except Exception as e:
            report["bridge"] = {"error": str(e)}
            report["step2_unit_tests"]["bridge_forward"] = {"pass": False, "error": str(e)}

        report["import_errors"] = dict(self._import_errors)
        return report


#=============================================================================
# CLI self-test
#=============================================================================
if __name__ == "__main__":
    connector = UniversalSESIConnector(build=True)
    report = connector.run_self_test(batch_size=2, device="cpu")

    print(json.dumps(report, indent=2, default=str))
