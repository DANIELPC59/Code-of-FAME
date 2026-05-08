# Code-of-FAME
Code od FAME (FAME: Few-Shot Adaptive Perturbation Generation for Scalable Website Fingerprinting Defense via Mixture-of-Experts)



# FAME

Reference implementation for **FAME: Few-Shot Adaptive Perturbation Generation for Scalable Website Fingerprinting Defense via Mixture-of-Experts**.

## Repository Layout

- `DataTool_Code/`: dataset loading and label utilities.
- `WF_Model/`: website-fingerprinting classifier models and training/evaluation scripts.
- `Defence_Method/FAME/`: FAME feature extractor, perturbation generator, and few-shot evaluation code.
- `Defence_Method/AWA/`, `Defence_Method/Alert/`, `Defence_Method/DFD/`, `Defence_Method/WalkieTalkie/`, `Defence_Method/Front_Base_Burst/`: baseline defense implementations.
- `Tool_Code/`: shared metrics and helper functions.

## Data And Model Paths

The code expects datasets and generated artifacts to follow the paths configured in the scripts, especially:

- `dataset/Website_Fingerprinting/...` for raw and processed datasets.
- `WF_Model/ModelSave/...` for trained classifier weights.
- `Defence_Method/*/File_Save/...` or `Defence_Method/*/FileSave/...` for generated defense artifacts.

If your local layout differs, update the path constants in the corresponding scripts before running experiments.

## Basic Workflow

1. Train or prepare website-fingerprinting classifiers with `WF_Model/Train_CFModel_NoDef.py`.
2. Pretrain the FAME feature extractor with `Defence_Method/FAME/Feature_Extracor/train.py`.
3. Train the FAME few-shot MoE perturbation generator with `Defence_Method/FAME/Noise_Gen/train_FewShot_MoeGen.py`.
4. Evaluate FAME and baseline defenses with the corresponding `Eva_*` or defense scripts.

## Notes

This repository is intended as an academic reference implementation. Large datasets, trained weights, generated perturbations, logs, and cached files are ignored by `.gitignore`.

# Deployment Prototypes

In addition , this repository includes two deployment-oriented prototypes for burst-sequence-based website-fingerprinting defense.

## P4_Switch

`P4_Switch/` contains the P4-switch deployment prototype. Its role is to move part of the defense logic into the programmable data plane, where packet/burst-level behavior can be handled close to the network path. The project is organized around P4 switch-side logic and the supporting control/runtime components needed to configure and test the defense behavior.

## PluggableTransport

`PluggableTransport/` contains the pluggable-transport deployment prototype. It implements the defense at the transport/proxy layer, making it suitable for integration with systems that support PT-style traffic transformation. Compared with the P4 version, this path keeps the defense in software and is easier to deploy and iterate without programmable-switch hardware.
