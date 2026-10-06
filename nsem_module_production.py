"""
nsem/module_production.py

# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026

"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .core import NSEMConfig, NeuroSensoryEnhancementModule
from .hardware.dac import DACSpec, NeuralStimulatorFrontend
from .hardware.chemical import MicrofluidicPulseModel
from .hardware.interlock import SynapticInterlock, SafetyEnvelope


@dataclass
class ProductionConfig:
    nsem: NSEMConfig
    dac: DACSpec
    implant_id: str = "NSEM-IMPLANT-001"
    interlock_secret: bytes = b"\x00" * 32


class ProductionNSEM(nn.Module):
    """
    Full production stack:

      spikes -> NSEM core -> motor_head -> DAC -> interlock -> electrodes
                            chemical_head -> microfluidic -> interlock -> apertures
                            vision_latent -> photonic HAL -> optical cortex map
                            cognitive_feedback -> quake HAL -> quantum co-processor
    """

    def __init__(self, cfg: ProductionConfig):
        super().__init__()
        self.cfg = cfg
        self.nsem = NeuroSensoryEnhancementModule(cfg.nsem)
        self.dac = NeuralStimulatorFrontend(cfg.dac)
        self.chemical = MicrofluidicPulseModel()
        self.interlock = SynapticInterlock(
            envelope=SafetyEnvelope(
                max_charge_uc_cm2=cfg.dac.max_charge_density_uc_cm2,
                max_current_ua=cfg.dac.full_scale_current_ua,
                max_pulse_width_us=cfg.dac.pulse_width_us,
                max_temperature_rise_c=1.0,
                allowed_electrodes=frozenset(range(cfg.dac.channels)),
                therapy_mode="sensory_restoration",
            ),
            implant_id=cfg.implant_id,
            secret=cfg.interlock_secret,
        )

    def forward(
        self,
        neural_spike_stream: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        patient_state: Optional[dict] = None,
        device_health: Optional[dict] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        # ---- 1. Core NSEM -----------------------------------------------
        outputs, next_memory = self.nsem(neural_spike_stream, memory_state)

        # ---- 2. Motor path -> DAC -> interlock --------------------------
        dac_out = self.dac(outputs["motor_control"])
        outputs["dac_cathodic_ua"] = dac_out["cathodic_ua"]
        outputs["dac_anodic_ua"] = dac_out["anodic_ua"]
        outputs["residual_charge_uc"] = dac_out["residual_charge_uc"]

        # ---- 3. Chemical path -> microfluidic -> interlock --------------
        chem_out = self.chemical(outputs["olfactory_gustatory"])
        outputs.update({f"chem_{k}": v for k, v in chem_out.items()})

        # ---- 4. Safety validation ---------------------------------------
        if patient_state is not None and device_health is not None:
            decision, receipt = self.interlock.validate(
                command={
                    "charge_uc_cm2": outputs["residual_charge_uc"].detach().max().item(),
                    "current_ua": dac_out["cathodic_ua"].detach().abs().max().item(),
                    "pulse_width_us": self.cfg.dac.pulse_width_us,
                    "electrodes": list(range(self.cfg.dac.channels)),
                },
                patient_state=patient_state,
                device_health=device_health,
            )
            outputs["interlock_decision"] = decision
            outputs["interlock_receipt"] = receipt
            if decision.value == "deny":
                # Hard-zero the output (safe fallback)
                outputs["dac_cathodic_ua"] = torch.zeros_like(dac_out["cathodic_ua"])
                outputs["dac_anodic_ua"] = torch.zeros_like(dac_out["anodic_ua"])
                outputs["chem_flow_nl_s"] = torch.zeros_like(chem_out["flow_nl_s"])

        return outputs, next_memory

    def parameter_groups(self, weight_decay: float = 0.01):
        return self.nsem.parameter_groups(weight_decay)
