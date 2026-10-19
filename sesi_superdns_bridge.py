"""
sesi_superdns_bridge.py
=============================================================================
SESI SUPER-DNS BRIDGE
Connects super_dns_one_v6_3-8.py directly into the SESI Universal Connector.

This bridge is more natural than the OpenFOAM bridge because SUPER DNS ONE
already produces PyTorch tensor fields, enabling direct GPU/differentiable
integration with SESI modules.

Important:
  SUPER DNS uses non-dimensional internal fields in several places.
  If feeding into SESI physics modules that expect SI units, set
  convert_to_si=True and provide appropriate scale factors.

Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
License   : MIT
Year      : 2026
=============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sesi_universal_connector import StandardSample
    _HAS_STANDARD_SAMPLE = True
except Exception:
    StandardSample = None
    _HAS_STANDARD_SAMPLE = False


class SuperDNSBridge:
    """
    Converts SUPER DNS ONE solver state into SESI StandardSample.

    Example:
        from super_dns_one_v6_3_8 import CFDConfig, CompressibleSolver

        cfg = CFDConfig(nx=32, ny=32, nz=32, device="cuda")
        solver = CompressibleSolver(cfg)
        solver.initialize("hypersonic_bnd")

        bridge = SuperDNSBridge(
            solver,
            convert_to_si=True,
            velocity_scale=340.0,
            temperature_scale=300.0,
            pressure_scale=101325.0,
            density_scale=1.225,
        )

        sample = bridge.to_standard_sample()
    """

    def __init__(
        self,
        solver: Any,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        convert_to_si: bool = True,
        velocity_scale: float = 1.0,
        temperature_scale: float = 300.0,
        pressure_scale: float = 101325.0,
        density_scale: float = 1.225,
        clamp_positive: bool = True,
    ) -> None:
        self.solver = solver
        self.dtype = dtype

        if device is None:
            self.device = getattr(solver, "device", torch.device("cpu"))
        else:
            self.device = torch.device(device)

        self.convert_to_si = convert_to_si
        self.velocity_scale = float(velocity_scale)
        self.temperature_scale = float(temperature_scale)
        self.pressure_scale = float(pressure_scale)
        self.density_scale = float(density_scale)
        self.clamp_positive = clamp_positive

    #----------------------------------------------------------------------
    def _clean(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return x

    #----------------------------------------------------------------------
    def get_primitive_fields(self) -> Dict[str, torch.Tensor]:
        """
        Returns primitive fields from SUPER DNS solver:
            rho, u, v, w, p, T, c, Ma
        """
        prim = self.solver.get_primitives()
        return {
            "rho": self._clean(prim["rho"]),
            "u": self._clean(prim["u"]),
            "v": self._clean(prim["v"]),
            "w": self._clean(prim["w"]),
            "p": self._clean(prim["p"]),
            "T": self._clean(prim["T"]),
            "c": self._clean(prim["c"]),
            "Ma": self._clean(prim["Ma"]),
        }

    #----------------------------------------------------------------------
    def to_standard_sample(self) -> Any:
        """
        Converts SUPER DNS fields into SESI StandardSample.

        Returned tensor shapes:
            velocity_field : [1, 3, D, H, W]
            thermal_field  : [1, 1, D, H, W]
            pressure_field : [1, 1, D, H, W]
            density_field  : [1, 1, D, H, W]
            em_field       : [1, 3, D, H, W]
            emf_coupling   : [1, 1, D, H, W]
            sigma_skin     : [1, 1, D, H, W]
        """
        prim = self.get_primitive_fields()

        u = prim["u"]
        v = prim["v"]
        w = prim["w"]

        # [D,H,W] -> [1,3,D,H,W]
        velocity_field = torch.stack([u, v, w], dim=0).unsqueeze(0)

        # [D,H,W] -> [1,1,D,H,W]
        thermal_field = prim["T"].unsqueeze(0).unsqueeze(0)
        pressure_field = prim["p"].unsqueeze(0).unsqueeze(0)
        density_field = prim["rho"].unsqueeze(0).unsqueeze(0)

        if self.convert_to_si:
            velocity_field = velocity_field * self.velocity_scale
            thermal_field = thermal_field * self.temperature_scale
            pressure_field = pressure_field * self.pressure_scale
            density_field = density_field * self.density_scale

        if self.clamp_positive:
            thermal_field = thermal_field.clamp_min(1.0)
            pressure_field = pressure_field.clamp_min(1.0)
            density_field = density_field.clamp_min(1.0e-6)

        em_field = torch.zeros_like(velocity_field)
        emf_coupling = torch.zeros_like(thermal_field)
        sigma_skin = torch.ones_like(thermal_field)

        if not _HAS_STANDARD_SAMPLE:
            return {
                "velocity_field": velocity_field,
                "thermal_field": thermal_field,
                "pressure_field": pressure_field,
                "density_field": density_field,
                "em_field": em_field,
                "emf_coupling": emf_coupling,
                "sigma_skin": sigma_skin,
            }

        return StandardSample(
            velocity_field=velocity_field,
            thermal_field=thermal_field,
            pressure_field=pressure_field,
            density_field=density_field,
            em_field=em_field,
            emf_coupling=emf_coupling,
            sigma_skin=sigma_skin,
        )

    #----------------------------------------------------------------------
    def to_npz(self, path: str) -> None:
        """
        Saves current SUPER DNS state in the same NPZ convention used by
        sesi_openfoam_bridge.py:

            U   : (D, H, W, 3)
            p   : (D, H, W)
            T   : (D, H, W)
            rho : (D, H, W)
        """
        sample = self.to_standard_sample()

        if isinstance(sample, dict):
            velocity = sample["velocity_field"]
            pressure = sample["pressure_field"]
            thermal = sample["thermal_field"]
            density = sample["density_field"]
        else:
            velocity = sample.velocity_field
            pressure = sample.pressure_field
            thermal = sample.thermal_field
            density = sample.density_field

        U = velocity[0].permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.float32)
        p = pressure[0, 0].detach().cpu().numpy().astype(np.float32)
        T = thermal[0, 0].detach().cpu().numpy().astype(np.float32)
        rho = density[0, 0].detach().cpu().numpy().astype(np.float32)

        np.savez_compressed(path, U=U, p=p, T=T, rho=rho)

    #----------------------------------------------------------------------
    def run_sesi_connector(
        self,
        connector: Optional[Any] = None,
        module_paths: Optional[Dict[str, str]] = None,
        build: bool = True,
    ) -> Dict[str, Any]:
        """
        Converts solver state and immediately runs it through
        UniversalSESIConnector.
        """
        sample = self.to_standard_sample()

        if connector is None:
            from sesi_universal_connector import UniversalSESIConnector

            connector = UniversalSESIConnector(
                module_paths=module_paths,
                build=build,
            )

        return connector(sample)
