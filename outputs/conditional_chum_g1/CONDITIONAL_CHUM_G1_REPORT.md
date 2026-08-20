# Conditional CHUM G1 Report

**Decision: PASS TO ARCHITECTURE GATE.** A ridge conditional imputer trained only on normal training segments predicted current XMV from current XMEAS and previous XMV. Replacing one XMV channel with this normal-conditional value removed all >0.005 test pre-FPR shifts seen under zero occlusion.

Material and 5/5 direction-stable AUROC losses remained for faults 4/XMV10 (0.4191), 19/XMV08 (0.0289), 25/XMV02 (0.0217), and 26/XMV04 (0.0840). The mean maximum conditional-replacement loss was 0.0798 for the frozen GAIN group and 0.0047 for no-gain faults.

Faults 7 and 24 did not retain a >=0.02 single-channel effect under conditional replacement. Their zero-occlusion channel results must not be used as primary attribution evidence.

This imputer estimates a conditional mean and does not represent a physical counterfactual controller trajectory. It is a more distribution-respecting attribution baseline, not a causal intervention.
