┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING (PyTorch)                           │
│  ProductionNSEM  ── AMP ── DDP ── gradient checkpointing            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  torch.onnx.export
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ONNX  (nsem.onnx, opset 18)                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  iree-opt / mlir-opt
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MLIR  (nsem.mlir)                                                  │
│   ├── linalg / tensor / arith   (hardware-agnostic)                 │
│   ├── quake dialect             (superconducting cognition)         │
│   ├── photonic custom dialect   (Xanadu / PsiQuantum)               │
│   └── dac dialect               (current-steering codes)            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  iree-compile
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VMFB artifacts                                                     │
│   nsem_photonic.vmfb   → optical cortex map                         │
│   nsem_superconducting.vmfb → quantum co-processor                  │
│   nsem_fpga.vmfb       → auditory DSP                               │
│   nsem_asic_dac.vmfb   → current-steering stimulator                │
│   nsem_chemical.vmfb   → microfluidic pulse controller              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  NSEMRuntime
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  IMPLANT HARDWARE                                                   │
│   64-ch AFE ── 4-ch HV stimulator ── microfluidic apertures         │
│   Secure interlock ── charge-balance monitor ── temp sensor         │
└─────────────────────────────────────────────────────────────────────┘
