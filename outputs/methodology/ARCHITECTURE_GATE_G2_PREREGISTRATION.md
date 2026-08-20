# Architecture Gate G2 — Frozen Execution Note

- Architectures: two-layer TCN and two-layer compact Transformer.
- Input window: 20; target: next-step 41 XMEAS.
- Conditions per architecture: F0 sensor-only, F1 sensor+XMV, F0-C capacity-matched sensor-only.
- Seeds: 42–46; epochs: 10; batch size: 1024; Adam learning rate: 0.001.
- Split, scaler, threshold percentile, alarm persistence, and test cohort are inherited unchanged from Experiment 1.
- TCN and Transformer F0-C widths are selected solely by closest parameter count to their F1 model before test evaluation.
- Primary robustness question: do the Experiment-1 fault-specific F1 gains survive both F0 and F0-C in at least one non-GRU family with direction agreement in 4/5 seeds?
- Strong architecture consensus requires support in GRU plus both TCN and Transformer. Support in GRU plus one family is partial consensus.
- Hyperparameters are frozen before the full five-seed results are inspected.
