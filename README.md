# SystemID

**Data-driven digital twin of a nonlinear, piezo-actuated electro-mechanical system**, built during a summer internship in the Control Systems Design department of ASELSAN's Microelectronics, Guidance & Electro-Optics division. The project compares three modeling approaches — a NARX-based MLP, a spectrogram/scalogram-driven CNN (ResNet-18 transfer learning), and a Laplace Neural Operator (LNO) — and selects the LNO as the final model, including a lightweight incremental version fast enough for closed-loop use in Simulink.

## Why

Testing controllers directly on the real hardware is slow to set up and financially risky — an unstable test can permanently damage expensive equipment. The system is also nonlinear: its response depends heavily on input amplitude, which rules out standard linear identification methods (ARX, transfer-function fitting). The goal of this project is a **digital twin** accurate enough for offline testing and future controller design, without needing access to the physical plant.

## Approach

Three architectures were implemented and compared end-to-end on real system data, each fixing a shortcoming of the one before it:

| # | Model | Idea | Outcome |
|---|-------|------|---------|
| 1 | **NARX–MLP** | PACF/cross-correlation feature selection feeds a 4-layer MLP (256–128–64–32, tanh, dropout) | Fast to train, but failed to generalize and underestimated overshoot |
| 2 | **CNN (ResNet-18)** | Input/output signals converted to STFT/CWT spectrogram-scalogram images; ResNet-18 fine-tuned (conv4–conv5) as an image regressor | Generalized well, but too slow for real-time, closed-loop use |
| 3 | **Laplace Neural Operator (LNO)** | 16 learnable damped-oscillatory poles + residue mixing of `u`/`y`, plus a pointwise local branch | ✅ Selected as the final model — accurate and fast enough for real-time use |

The final LNO was additionally converted into an **incremental, O(1)-per-step predictor** (`laplaceFastPredictor`) for real-time closed-loop simulation in Simulink, updating its pole coefficients sample-by-sample instead of reprocessing the full input window.

All three approaches were evaluated with both open-loop (one-step-ahead) and **closed-loop** testing, where the model has no access to ground-truth output and must run purely on its own past predictions — mirroring how a digital twin would actually be used.

### Results

Real-system results are confidential to ASELSAN. For a public, reproducible reference, the same pipelines were also run on the [Wiener–Hammerstein benchmark](https://www.nonlinearbenchmark.org/):

| Model | Fit % (benchmark) |
|---|---|
| NARX–MLP | 96.24% |
| CNN / ResNet-18 | −17.0% (benchmark set is far smaller than the real dataset that data-hungry CNNs need; it performed much better on the real data) |
| **LNO (final model)** | **95.44%** |

On the real, unseen system data, the closed-loop LNO pipeline reached **80%+ fit accuracy**. The main remaining limitation is resonance behavior, likely due to limited frequency diversity in the training data (see *Future Work*).

## Repository structure

```
SystemID/
├── scripts/                     # Top-level entry points — run these
│   ├── MAIN_WORKFLOW.m           # parse → feature extraction → train MLP → predict
│   ├── PREDICTION_WORKFLOW.m     # parse test data → Simulink prep → closed-loop MLP predict
│   └── LNO_predict_workflow.m    # parse test data → Simulink prep (LNO) → closed-loop LNO predict + compare
├── src/
│   ├── parse/                   # Raw Speedgoat/Simulink logs → clean input/output vectors
│   │   ├── parse_data.m
│   │   └── parse_test_data.m
│   ├── signal_gen/               # Synthetic excitation signal generators
│   │   ├── APRBS.m                  # Amplitude-modulated PRBS (primary training input)
│   │   ├── frf_chirp.m              # Swept-sine chirp for frequency response estimation
│   │   └── square_wave.m            # Step response tests (overshoot, settling time)
│   ├── features/                 # Feature extraction & frequency-domain representations
│   │   ├── feature_extraction.m     # PACF + cross-correlation → NARX regressor lags
│   │   ├── Spectrogram.m            # STFT image generation
│   │   ├── Scalogram.m              # CWT image generation
│   │   └── SINDy_demo.m
│   ├── models/                   # Model definitions
│   │   ├── ARX_model.m              # Linear baseline (System Identification Toolbox)
│   │   ├── multi_data_MLP.m         # NARX-based MLP
│   │   └── LSTM.m
│   ├── train/                    # Training scripts
│   │   ├── laplace_network_train.m  # LNO training
│   │   └── multiple_train_cnn.m     # ResNet-18 fine-tuning per dataset
│   ├── predict/                  # Open/closed-loop prediction & evaluation
│   │   ├── mlp_predict.m
│   │   ├── cnn_prediction.m
│   │   ├── simulink_predict.m
│   │   └── lno_simulink_predict.m
│   ├── blind_predict/
│   │   └── blind_predict.m          # Closed-loop LNO prediction on a user-selected dataset
│   ├── simulink_models/          # .mdl closed-loop simulation models
│   │   ├── closed_loop_model.mdl
│   │   └── cnn_closed_loop_model.mdl
│   └── helpers/
│       ├── laplaceLayer.m           # Custom LNO layer (pole-residue neural operator)
│       ├── laplaceFastPredictor.m   # Incremental O(1)-per-step LNO predictor (Simulink System block)
│       ├── extractLaplaceParams.m   # Trained LNO → lightweight struct for streaming inference
│       ├── ResNet16TransferLearning.m
│       ├── SimulinkDataPrep.m
│       ├── simulinkDataPrepLno       # LNO variant of the above
│       ├── loadStats.m
│       └── log_run.m                # Archives each training/eval run (model + figures) under data/results/
├── Change Logs/
│   └── ImageAnalysisChanges.txt  # Iteration log for the CNN/ResNet-18 fine-tuning process
└── .gitignore                    # Excludes *.mat/*.csv data, MEX/SLX build artifacts, etc.
```

## Getting started

**Requirements:** MATLAB with the Deep Learning Toolbox, System Identification Toolbox, Signal Processing Toolbox, and Simulink.

1. Open the project root in MATLAB.
2. Populate `data/` (gitignored — see *Data* below).
3. Run one of the top-level workflows from `scripts/`:

   ```matlab
   MAIN_WORKFLOW          % end-to-end: parse data, extract features, train the MLP, predict
   PREDICTION_WORKFLOW    % closed-loop evaluation of a trained MLP in Simulink
   LNO_predict_workflow   % closed-loop evaluation of the trained LNO in Simulink
   ```

   Each script sets up `addpath(genpath("src"))` and `cd`s to the project root automatically, so they can be run from any working folder.

### Data

Raw and processed data (`data/`), trained models, and generated spectrogram/scalogram images are **not included** in this repository — they are excluded via `.gitignore` and, for the real ASELSAN system, are confidential. To reproduce the pipeline you'll need your own `data/raw/`, `data/raw/test/` folders in the same layout the parse scripts expect, or the public [Wiener–Hammerstein benchmark](https://www.nonlinearbenchmark.org/) used for the reported benchmark numbers above. `SimulinkDataPrep.m` also expects external controller `.mat` files (e.g. `Controllers/controller_base.mat`) that are specific to the target hardware and are likewise not included.

## Limitations & future work

- The model still struggles to reproduce **resonance behavior**, most likely because the training data didn't cover enough frequency diversity to excite it.
- Retraining the LNO on more frequency-diverse excitation signals is the most direct next step.
- This project deliberately stops at system identification — no controller was designed against the model. A natural extension is to use the trained (incremental) LNO as the plant model in a model-in-the-loop controller design and testing workflow.

## Tools

MATLAB · Simulink · Deep Learning Toolbox · System Identification Toolbox · Signal Processing Toolbox · Git

## Acknowledgements

This project was completed during an internship (22.06.2026 – 03.08.2026) in ASELSAN's Microelectronics, Guidance & Electro-Optics division, under the supervision of Ata Köklü, alongside project partner Selim Efe Aytemur. Thanks to the control systems design department and the ASELSAN team for their guidance and for making real-system testing time available throughout the internship.

**Author:** Arda Gencer, Computer Science and Engineering / Mechatronics Engineering, Sabancı University
