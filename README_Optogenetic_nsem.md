                    ┌──────────────────────┐
                    │   Controller / Loss  │
                    └──────┬───────────────┘
                           │ irradiance I(x,λ,t)
                           ▼
                ┌──────────────────────┐
                │ AdvancedOptogenetic  │
                │      Engine          │
                └──────┬───────────────┘
                       │ I_photo [B,N]
                       │
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
│ NSEM Coupling│  │              │  │ Phototoxicity   │
│  (neural     │  │              │  │ Safety Layer    │
│   mass)      │  │              │  │                 │
└──────┬───────┘  │              │  │ ∫I dt → fluence │
       │          │              │  │ Arrhenius Ω     │
       │ V_mem    │              │  │ No-Zeno chatter │
       └──────────┴──────────────┘  │ double-exp      │
         (feedback loop)           │ barrier         │
                                   └────────┬────────┘
                                            │ safety_loss
                                            │ margin ∈ [0,1]
                                            ▼
                                    (add to training loss)
