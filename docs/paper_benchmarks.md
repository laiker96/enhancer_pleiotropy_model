# Paper model benchmarks

Frozen reference measurements from the current transfer-learning comparison: **7 regressors and 87 classifiers**. These are historical results, not measurements from a fresh portable-workflow run.

The [machine-readable snapshot](../reproduce/benchmarks/models.json) retains all recorded validation/test metrics, per-context results, learning rates, training settings, parent lineage, relative artifact paths and full checkpoint SHA256 hashes. Missing values are null/—, not zeros.

## How to read these tables

- **E:** active-in-context enhancers versus other catalog enhancers. **E+BG:** active enhancers versus other enhancers plus matched genomic background. **A/BG:** active enhancers versus their matched background sequences only. AP means average precision, not trapezoidal PR area; it depends on class prevalence.
- Classifier tables report mean ± sample SD across three classifier seeds, not confidence intervals. Fine-tuning seeds share one pretrained regressor per architecture.
- Every checkpoint was selected on validation, not test. Classifier epoch selection: maximum validation enhancer-only macro AP, BCE tie-break. The representative seed shown below maximizes validation E+BG AP for background-trained models, otherwise validation E AP.
- Validation is the first half of chr2L; test is chr3R. This historical test set has been inspected repeatedly and is not a new untouched holdout. No superiority test is claimed.
- Replacement and retained hidden readouts differ in architecture as well as transferred weights. Scratch initializes the same corresponding classifier architecture randomly; fine-tuning updates all transferred layers, not a frozen encoder.

## Recommended historical checkpoints

| Purpose | Checkpoint | Epoch | Val E AP | Val E+BG AP | Test E AP | Test E+BG AP |
| --- | --- | --- | --- | --- | --- | --- |
| Enhancer-only discrimination | `cnn_finetune_20260914` | 38 | 0.667 | — | 0.653 | 0.515 |
| Enhancer + background; calibrated IG analysis | `background__cnn_finetune_20260914` | 38 | 0.649 | 0.586 | 0.637 | 0.583 |

These single-checkpoint scores are not seed averages. A model that wins for E need not win for E+BG. Use the latter checkpoint for the current calibrated-attribution workflow.

## Primary architectures: enhancer-only training

| Ref | Model | Hidden policy | Initialization | Train population | Val E AP | Val E+BG AP | Test E AP | Test E+BG AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | CNN + attention | Replace assay hidden | Fine-tune (2048 bp parent) | E | 0.6294 ± 0.0002 | — | 0.6165 ± 0.0004 | 0.4840 ± 0.0010 |
| C02 | CNN + attention | Replace assay hidden | Scratch | E | 0.5370 ± 0.0031 | — | 0.5233 ± 0.0037 | 0.3538 ± 0.0073 |
| C04 | CNN + attention | Keep assay hidden | Fine-tune (2048 bp parent) | E | 0.6287 ± 0.0006 | — | 0.6166 ± 0.0002 | — |
| C06 | CNN + attention | Keep assay hidden | Scratch | E | 0.5391 ± 0.0014 | — | 0.5226 ± 0.0009 | — |
| C07 | Dilated CNN | Replace assay hidden | Fine-tune (2048 bp parent) | E | 0.6664 ± 0.0002 | — | 0.6531 ± 0.0004 | 0.5150 ± 0.0007 |
| C08 | Dilated CNN | Replace assay hidden | Scratch | E | 0.5763 ± 0.0050 | — | 0.5682 ± 0.0056 | 0.4084 ± 0.0079 |
| C10 | Dilated CNN | Keep assay hidden | Fine-tune (2048 bp parent) | E | 0.6569 ± 0.0006 | — | 0.6463 ± 0.0003 | — |
| C12 | Dilated CNN | Keep assay hidden | Scratch | E | 0.5801 ± 0.0047 | — | 0.5676 ± 0.0033 | — |
| C19 | Flatten/dense CNN | Keep shared dense | Fine-tune (2048 bp parent) | E | 0.6393 ± 0.0009 | — | 0.6247 ± 0.0005 | — |
| C21 | Flatten/dense CNN | Keep shared dense | Scratch | E | 0.4858 ± 0.0051 | — | 0.4792 ± 0.0026 | — |

## Primary architectures: 25% background training

| Ref | Model | Hidden policy | Initialization | Train population | Val E AP | Val E+BG AP | Test E AP | Test E+BG AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C03 | CNN + attention | Keep assay hidden | Fine-tune (2048 bp parent) | E + BG | 0.6201 ± 0.0010 | 0.5480 ± 0.0007 | 0.6073 ± 0.0012 | 0.5399 ± 0.0014 |
| C05 | CNN + attention | Keep assay hidden | Scratch | E + BG | 0.5338 ± 0.0008 | 0.4389 ± 0.0053 | 0.5227 ± 0.0017 | 0.4299 ± 0.0049 |
| C09 | Dilated CNN | Keep assay hidden | Fine-tune (2048 bp parent) | E + BG | 0.6480 ± 0.0007 | 0.5852 ± 0.0010 | 0.6373 ± 0.0003 | 0.5829 ± 0.0002 |
| C11 | Dilated CNN | Keep assay hidden | Scratch | E + BG | 0.5773 ± 0.0056 | 0.5030 ± 0.0077 | 0.5664 ± 0.0040 | 0.4974 ± 0.0063 |
| C18 | Flatten/dense CNN | Keep shared dense | Fine-tune (2048 bp parent) | E + BG | 0.6307 ± 0.0011 | 0.5842 ± 0.0013 | 0.6171 ± 0.0008 | 0.5740 ± 0.0008 |
| C20 | Flatten/dense CNN | Keep shared dense | Scratch | E + BG | 0.4796 ± 0.0047 | 0.4171 ± 0.0086 | 0.4714 ± 0.0049 | 0.4072 ± 0.0083 |

## Additional historical architectures and shorter-input pilots

These results are retained for completeness. EnhancerNet and 512/1,024-bp runs are **not** supported by the portable training dispatcher. The 512-bp classifiers ran only 15 epochs of the original 40-epoch LR curve; their test metrics were not measured. Short-input regressors also changed H3K27ac supervision to 512 bp, so this is not a pure input-length comparison.

| Ref | Model | Hidden policy | Initialization | Train population | Val E AP | Val E+BG AP | Test E AP | Test E+BG AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C13 | Dilated CNN 1,024 | Keep assay hidden | Fine-tune (1024 bp parent) | E + BG | 0.6323 ± 0.0006 | 0.5704 ± 0.0004 | 0.6228 ± 0.0003 | 0.5696 ± 0.0006 |
| C14 | Dilated CNN 512 | Keep assay hidden | Fine-tune (1024 bp parent) | E + BG | 0.6154 ± 0.0003 | 0.5520 ± 0.0010 | — | — |
| C15 | Dilated CNN 512 | Keep assay hidden | Fine-tune (2048 bp parent) | E + BG | 0.6167 ± 0.0009 | 0.5510 ± 0.0020 | — | — |
| C16 | Dilated CNN 512 | Keep assay hidden | Fine-tune (512 bp parent) | E + BG | 0.6013 ± 0.0006 | 0.5420 ± 0.0004 | — | — |
| C17 | Dilated CNN 512 | Keep assay hidden | Scratch | E + BG | 0.5228 ± 0.0024 | 0.4562 ± 0.0014 | — | — |
| C22 | EnhancerNet + attention | Keep shared dense | Fine-tune (2048 bp parent) | E + BG | 0.5533 ± 0.0002 | 0.5100 ± 0.0003 | 0.5398 ± 0.0002 | 0.5001 ± 0.0005 |
| C23 | EnhancerNet + attention | Keep shared dense | Fine-tune (2048 bp parent) | E | 0.5638 ± 0.0006 | — | 0.5498 ± 0.0004 | — |
| C24 | EnhancerNet + attention | Keep shared dense | Scratch | E + BG | 0.4638 ± 0.0044 | 0.4028 ± 0.0042 | 0.4508 ± 0.0051 | 0.3921 ± 0.0044 |
| C25 | EnhancerNet + attention | Keep shared dense | Scratch | E | 0.4743 ± 0.0041 | — | 0.4584 ± 0.0049 | — |
| C26 | EnhancerNet CNN | Keep shared dense | Fine-tune (2048 bp parent) | E + BG | 0.4039 ± 0.0001 | 0.3287 ± 0.0002 | 0.3958 ± 0.0004 | 0.3240 ± 0.0004 |
| C27 | EnhancerNet CNN | Keep shared dense | Fine-tune (2048 bp parent) | E | 0.4129 ± 0.0004 | — | 0.4054 ± 0.0004 | — |
| C28 | EnhancerNet CNN | Keep shared dense | Scratch | E + BG | 0.3606 ± 0.0034 | 0.2750 ± 0.0048 | 0.3545 ± 0.0025 | 0.2727 ± 0.0035 |
| C29 | EnhancerNet CNN | Keep shared dense | Scratch | E | 0.3848 ± 0.0047 | — | 0.3760 ± 0.0078 | — |

## Background rejection and enhancer discrimination

Same seeds and selected checkpoints as above; all values are three-seed mean ± SD. The FPR column is a diagnostic threshold at 80% enhancer recall, not a deployment cutoff.

| Ref | Val E AUROC | Val E Brier | Val A/BG AP | Test A/BG AP | Val BG FPR@80% recall |
| --- | --- | --- | --- | --- | --- |
| C01 | 0.8550 ± 0.0003 | 0.1249 ± 0.0001 | — | 0.8597 ± 0.0006 | — |
| C02 | 0.7996 ± 0.0024 | 0.1420 ± 0.0008 | — | 0.7639 ± 0.0076 | — |
| C03 | 0.8518 ± 0.0006 | 0.1269 ± 0.0009 | 0.9243 ± 0.0016 | 0.9240 ± 0.0010 | 0.0965 ± 0.0018 |
| C04 | 0.8555 ± 0.0006 | 0.1254 ± 0.0005 | — | — | — |
| C05 | 0.7998 ± 0.0025 | 0.1431 ± 0.0009 | 0.8638 ± 0.0054 | 0.8649 ± 0.0050 | 0.2012 ± 0.0080 |
| C06 | 0.8016 ± 0.0007 | 0.1427 ± 0.0013 | — | — | — |
| C07 | 0.8674 ± 0.0002 | 0.1183 ± 0.0000 | — | 0.8668 ± 0.0009 | — |
| C08 | 0.8201 ± 0.0025 | 0.1366 ± 0.0013 | — | 0.7985 ± 0.0038 | — |
| C09 | 0.8611 ± 0.0002 | 0.1220 ± 0.0001 | 0.9335 ± 0.0004 | 0.9403 ± 0.0002 | 0.0832 ± 0.0012 |
| C10 | 0.8649 ± 0.0002 | 0.1195 ± 0.0000 | — | — | — |
| C11 | 0.8213 ± 0.0010 | 0.1368 ± 0.0015 | 0.9029 ± 0.0038 | 0.9051 ± 0.0032 | 0.1314 ± 0.0047 |
| C12 | 0.8213 ± 0.0022 | 0.1359 ± 0.0014 | — | — | — |
| C13 | 0.8528 ± 0.0001 | 0.1248 ± 0.0001 | 0.9297 ± 0.0004 | 0.9365 ± 0.0004 | 0.0878 ± 0.0003 |
| C14 | 0.8421 ± 0.0006 | 0.1282 ± 0.0001 | 0.9254 ± 0.0014 | — | 0.0972 ± 0.0024 |
| C15 | 0.8440 ± 0.0000 | 0.1275 ± 0.0004 | 0.9250 ± 0.0011 | — | 0.0970 ± 0.0018 |
| C16 | 0.8366 ± 0.0002 | 0.1300 ± 0.0001 | 0.9257 ± 0.0007 | — | 0.0961 ± 0.0013 |
| C17 | 0.7839 ± 0.0026 | 0.1458 ± 0.0008 | 0.8957 ± 0.0026 | — | 0.1447 ± 0.0043 |
| C18 | 0.8504 ± 0.0005 | 0.1256 ± 0.0002 | 0.9435 ± 0.0003 | 0.9466 ± 0.0002 | 0.0757 ± 0.0020 |
| C19 | 0.8548 ± 0.0005 | 0.1238 ± 0.0003 | — | — | — |
| C20 | 0.7499 ± 0.0021 | 0.1542 ± 0.0019 | 0.8842 ± 0.0075 | 0.8802 ± 0.0073 | 0.1758 ± 0.0127 |
| C21 | 0.7585 ± 0.0030 | 0.1514 ± 0.0010 | — | — | — |
| C22 | 0.7993 ± 0.0002 | 0.1381 ± 0.0001 | 0.9294 ± 0.0006 | 0.9326 ± 0.0007 | 0.0954 ± 0.0015 |
| C23 | 0.8083 ± 0.0004 | 0.1361 ± 0.0001 | — | — | — |
| C24 | 0.7353 ± 0.0026 | 0.1533 ± 0.0008 | 0.8840 ± 0.0022 | 0.8812 ± 0.0007 | 0.1885 ± 0.0050 |
| C25 | 0.7477 ± 0.0030 | 0.1504 ± 0.0011 | — | — | — |
| C26 | 0.7056 ± 0.0003 | 0.1606 ± 0.0000 | 0.8343 ± 0.0002 | 0.8413 ± 0.0005 | 0.3096 ± 0.0024 |
| C27 | 0.7163 ± 0.0002 | 0.1588 ± 0.0000 | — | — | — |
| C28 | 0.6630 ± 0.0036 | 0.1672 ± 0.0005 | 0.7957 ± 0.0049 | 0.7936 ± 0.0045 | 0.3410 ± 0.0098 |
| C29 | 0.6857 ± 0.0098 | 0.1639 ± 0.0013 | — | — | — |

## Validation breadth and context discrimination

Breadth compares the sum of the original, uncalibrated classifier probabilities with observed catalog degree of pleiotropy. These are **not** the subsequently fitted calibrated-probability metrics. Close-context accuracy uses discordant labels within related context pairs; family hit is the any-active-member top-family diagnostic. All columns are enhancer-only, three-seed mean ± SD.

| Ref | Breadth r | Breadth ρ | Breadth R² | Breadth MAE | Close-context accuracy | Family hit |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | 0.6056 ± 0.0005 | 0.5202 ± 0.0013 | 0.3546 ± 0.0011 | 0.7666 ± 0.0011 | 0.6836 ± 0.0037 | 0.7764 ± 0.0020 |
| C02 | 0.4744 ± 0.0154 | 0.4134 ± 0.0112 | 0.2189 ± 0.0132 | 0.9018 ± 0.0056 | 0.6561 ± 0.0036 | 0.7235 ± 0.0052 |
| C03 | 0.6088 ± 0.0023 | 0.5221 ± 0.0029 | 0.3361 ± 0.0051 | 0.8163 ± 0.0042 | 0.6832 ± 0.0029 | 0.7763 ± 0.0020 |
| C04 | 0.6133 ± 0.0006 | 0.5185 ± 0.0032 | 0.3653 ± 0.0008 | 0.7728 ± 0.0035 | 0.6838 ± 0.0018 | 0.7774 ± 0.0021 |
| C05 | 0.4828 ± 0.0065 | 0.4262 ± 0.0098 | 0.1928 ± 0.0120 | 0.9076 ± 0.0031 | 0.6551 ± 0.0072 | 0.7263 ± 0.0036 |
| C06 | 0.4783 ± 0.0034 | 0.4119 ± 0.0028 | 0.2121 ± 0.0111 | 0.9209 ± 0.0223 | 0.6483 ± 0.0049 | 0.7203 ± 0.0032 |
| C07 | 0.6616 ± 0.0011 | 0.5452 ± 0.0019 | 0.4340 ± 0.0019 | 0.7285 ± 0.0003 | 0.6961 ± 0.0028 | 0.7907 ± 0.0018 |
| C08 | 0.5371 ± 0.0085 | 0.4555 ± 0.0098 | 0.2823 ± 0.0105 | 0.8458 ± 0.0112 | 0.6571 ± 0.0062 | 0.7443 ± 0.0011 |
| C09 | 0.6432 ± 0.0007 | 0.5425 ± 0.0004 | 0.3859 ± 0.0015 | 0.7972 ± 0.0007 | 0.6990 ± 0.0023 | 0.7899 ± 0.0008 |
| C10 | 0.6459 ± 0.0019 | 0.5341 ± 0.0013 | 0.4143 ± 0.0028 | 0.7502 ± 0.0009 | 0.7004 ± 0.0015 | 0.7867 ± 0.0004 |
| C11 | 0.5462 ± 0.0038 | 0.4722 ± 0.0012 | 0.2613 ± 0.0040 | 0.8659 ± 0.0082 | 0.6675 ± 0.0006 | 0.7487 ± 0.0027 |
| C12 | 0.5310 ± 0.0075 | 0.4416 ± 0.0084 | 0.2706 ± 0.0118 | 0.8559 ± 0.0138 | 0.6643 ± 0.0033 | 0.7491 ± 0.0042 |
| C13 | 0.6271 ± 0.0005 | 0.5328 ± 0.0008 | 0.3625 ± 0.0024 | 0.8005 ± 0.0001 | 0.6927 ± 0.0031 | 0.7793 ± 0.0022 |
| C14 | 0.6060 ± 0.0005 | 0.5141 ± 0.0013 | 0.3430 ± 0.0040 | 0.8119 ± 0.0037 | 0.6796 ± 0.0018 | 0.7568 ± 0.0007 |
| C15 | 0.6120 ± 0.0022 | 0.5214 ± 0.0002 | 0.3522 ± 0.0047 | 0.8118 ± 0.0019 | 0.6857 ± 0.0029 | 0.7682 ± 0.0008 |
| C16 | 0.6061 ± 0.0008 | 0.5222 ± 0.0008 | 0.3500 ± 0.0010 | 0.8171 ± 0.0033 | 0.6685 ± 0.0026 | 0.7550 ± 0.0022 |
| C17 | 0.4831 ± 0.0065 | 0.4164 ± 0.0027 | 0.2047 ± 0.0068 | 0.9209 ± 0.0148 | 0.6514 ± 0.0023 | 0.6992 ± 0.0041 |
| C18 | 0.6337 ± 0.0018 | 0.5210 ± 0.0006 | 0.3653 ± 0.0021 | 0.8043 ± 0.0018 | 0.6695 ± 0.0010 | 0.7814 ± 0.0012 |
| C19 | 0.6340 ± 0.0023 | 0.5101 ± 0.0018 | 0.3920 ± 0.0031 | 0.7572 ± 0.0027 | 0.6690 ± 0.0042 | 0.7784 ± 0.0010 |
| C20 | 0.4501 ± 0.0033 | 0.3638 ± 0.0096 | 0.1591 ± 0.0121 | 0.9287 ± 0.0095 | 0.6137 ± 0.0065 | 0.6758 ± 0.0037 |
| C21 | 0.4534 ± 0.0035 | 0.3603 ± 0.0057 | 0.2004 ± 0.0037 | 0.9221 ± 0.0132 | 0.6060 ± 0.0061 | 0.6835 ± 0.0032 |
| C22 | 0.6194 ± 0.0010 | 0.5235 ± 0.0006 | 0.3727 ± 0.0020 | 0.7968 ± 0.0016 | 0.6330 ± 0.0011 | 0.7230 ± 0.0010 |
| C23 | 0.6285 ± 0.0010 | 0.5194 ± 0.0008 | 0.3937 ± 0.0013 | 0.7986 ± 0.0008 | 0.6326 ± 0.0018 | 0.7245 ± 0.0020 |
| C24 | 0.4700 ± 0.0056 | 0.4003 ± 0.0008 | 0.1833 ± 0.0166 | 0.9612 ± 0.0261 | 0.6069 ± 0.0039 | 0.6657 ± 0.0068 |
| C25 | 0.4740 ± 0.0105 | 0.3980 ± 0.0032 | 0.2150 ± 0.0120 | 0.9628 ± 0.0124 | 0.6107 ± 0.0040 | 0.6807 ± 0.0072 |
| C26 | 0.3937 ± 0.0001 | 0.3462 ± 0.0005 | 0.1366 ± 0.0003 | 0.9124 ± 0.0008 | 0.6053 ± 0.0007 | 0.6306 ± 0.0016 |
| C27 | 0.3976 ± 0.0006 | 0.3470 ± 0.0005 | 0.1564 ± 0.0002 | 0.9779 ± 0.0016 | 0.6062 ± 0.0009 | 0.6354 ± 0.0013 |
| C28 | 0.2883 ± 0.0014 | 0.2733 ± 0.0019 | 0.0634 ± 0.0029 | 0.9554 ± 0.0028 | 0.5975 ± 0.0026 | 0.5987 ± 0.0070 |
| C29 | 0.3005 ± 0.0276 | 0.2571 ± 0.0262 | 0.0812 ± 0.0162 | 1.0164 ± 0.0099 | 0.6025 ± 0.0037 | 0.6183 ± 0.0046 |

## Representative checkpoint settings

IDs refer to the validation-selected representative seed per condition, not the best test seed. Encoder/head columns show **peak** LR. Full selected-epoch and final LRs are in the snapshot. All classifiers use BCE; scratch uses 1e-4, fine-tuning 1e-5 for transferred layers and 1e-4 for the new head. The schedule is one-epoch warm-up then cosine to 10% of peak.

| Ref | Representative model ID | Selected / completed epochs | Peak LR encoder / head | Checkpoint SHA256 (prefix) |
| --- | --- | --- | --- | --- |
| C01 | `attention_finetune_20260914` | 13 / 40 | 1e-05 / 1e-04 | `b093fa84f433` |
| C02 | `attention_scratch_20260915` | 40 / 40 | 1e-04 / 1e-04 | `b96dc2580153` |
| C03 | `background__attention_finetune_20260915` | 12 / 40 | 1e-05 / 1e-04 | `db17bb23e961` |
| C04 | `retained_heads__attention_finetune_20260916` | 15 / 40 | 1e-05 / 1e-04 | `93ad6c4fc406` |
| C05 | `background__attention_scratch_20260915` | 38 / 40 | 1e-04 / 1e-04 | `8472f2e7e413` |
| C06 | `retained_heads__attention_scratch_20260916` | 39 / 40 | 1e-04 / 1e-04 | `dbe7f0ebb832` |
| C07 | `cnn_finetune_20260914` | 38 / 40 | 1e-05 / 1e-04 | `c6dddf28f802` |
| C08 | `cnn_scratch_20260916` | 36 / 40 | 1e-04 / 1e-04 | `0fb14e15fc85` |
| C09 | `background__cnn_finetune_20260914` | 38 / 40 | 1e-05 / 1e-04 | `7dc8eb721fbd` |
| C10 | `retained_heads__cnn_finetune_20260914` | 38 / 40 | 1e-05 / 1e-04 | `c00e680b2437` |
| C11 | `background__cnn_scratch_20260916` | 40 / 40 | 1e-04 / 1e-04 | `a52cb9795eb2` |
| C12 | `retained_heads__cnn_scratch_20260916` | 37 / 40 | 1e-04 / 1e-04 | `31719df514cc` |
| C13 | `cnn_1024_finetune_20260915` | 37 / 40 | 1e-05 / 1e-04 | `94eb88e0160a` |
| C14 | `cnn_512_from1024_20260914` | 15 / 15 | 1e-05 / 1e-04 | `eaa29e88c741` |
| C15 | `cnn_512_from2048_20260914` | 15 / 15 | 1e-05 / 1e-04 | `92afa6b65cbb` |
| C16 | `cnn_512_from512_20260915` | 15 / 15 | 1e-05 / 1e-04 | `34b8d5d14c2a` |
| C17 | `cnn_512_scratch_20260914` | 15 / 15 | 1e-04 / 1e-04 | `d2bcdeb99f88` |
| C18 | `background__dense_finetune_20260916` | 36 / 40 | 1e-05 / 1e-04 | `43f0536823c2` |
| C19 | `dense_finetune_20260916` | 36 / 40 | 1e-05 / 1e-04 | `12b761323db4` |
| C20 | `background__dense_scratch_20260915` | 9 / 40 | 1e-04 / 1e-04 | `f09e8061a6fd` |
| C21 | `dense_scratch_20260915` | 9 / 40 | 1e-04 / 1e-04 | `a5ac744ae5a9` |
| C22 | `background__enhancernet_attention_finetune_20260914` | 39 / 40 | 1e-05 / 1e-04 | `a08e6480103c` |
| C23 | `enhancernet_attention_finetune_20260915` | 40 / 40 | 1e-05 / 1e-04 | `f4a46f64ebc0` |
| C24 | `background__enhancernet_attention_scratch_20260915` | 40 / 40 | 1e-04 / 1e-04 | `51fabf2f1eb2` |
| C25 | `enhancernet_attention_scratch_20260915` | 40 / 40 | 1e-04 / 1e-04 | `a9cf8e8302ec` |
| C26 | `background__enhancernet_cnn_finetune_20260916` | 38 / 40 | 1e-05 / 1e-04 | `ecac42082f4d` |
| C27 | `enhancernet_cnn_finetune_20260916` | 38 / 40 | 1e-05 / 1e-04 | `ca39a62a0f9c` |
| C28 | `background__enhancernet_cnn_scratch_20260915` | 37 / 40 | 1e-04 / 1e-04 | `0633ccbb0ff3` |
| C29 | `enhancernet_cnn_scratch_20260915` | 40 / 40 | 1e-04 / 1e-04 | `5ebb6cabfdd9` |

## Validation AP per context: representative checkpoints

Enhancer-only evaluation; values are for the same representative checkpoint selected above. `ab` adult brain, `e13/e5` embryo stages, `ead/hid/wid` eye-antennal/haltere/wing discs, `lb` larval brain, `o` ovary.

| Ref | ab | e13 | e5 | ead | hid | lb | o | wid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 0.706 | 0.562 | 0.509 | 0.625 | 0.642 | 0.688 | 0.658 | 0.647 |
| C02 | 0.643 | 0.443 | 0.383 | 0.534 | 0.562 | 0.584 | 0.605 | 0.571 |
| C03 | 0.675 | 0.565 | 0.485 | 0.627 | 0.647 | 0.671 | 0.635 | 0.649 |
| C04 | 0.704 | 0.563 | 0.496 | 0.630 | 0.647 | 0.687 | 0.655 | 0.652 |
| C05 | 0.621 | 0.435 | 0.377 | 0.551 | 0.567 | 0.557 | 0.599 | 0.569 |
| C06 | 0.638 | 0.460 | 0.390 | 0.547 | 0.553 | 0.581 | 0.599 | 0.555 |
| C07 | 0.724 | 0.587 | 0.556 | 0.661 | 0.682 | 0.713 | 0.714 | 0.695 |
| C08 | 0.656 | 0.489 | 0.435 | 0.585 | 0.599 | 0.622 | 0.659 | 0.610 |
| C09 | 0.687 | 0.582 | 0.532 | 0.655 | 0.674 | 0.690 | 0.683 | 0.687 |
| C10 | 0.721 | 0.581 | 0.542 | 0.652 | 0.672 | 0.703 | 0.702 | 0.687 |
| C11 | 0.665 | 0.503 | 0.411 | 0.597 | 0.608 | 0.631 | 0.623 | 0.621 |
| C12 | 0.676 | 0.495 | 0.423 | 0.593 | 0.600 | 0.633 | 0.646 | 0.612 |
| C13 | 0.677 | 0.570 | 0.489 | 0.646 | 0.669 | 0.672 | 0.661 | 0.676 |
| C14 | 0.658 | 0.555 | 0.459 | 0.632 | 0.652 | 0.654 | 0.649 | 0.662 |
| C15 | 0.649 | 0.563 | 0.478 | 0.634 | 0.653 | 0.647 | 0.660 | 0.660 |
| C16 | 0.635 | 0.542 | 0.442 | 0.627 | 0.646 | 0.640 | 0.634 | 0.648 |
| C17 | 0.581 | 0.415 | 0.359 | 0.557 | 0.568 | 0.553 | 0.569 | 0.574 |
| C18 | 0.682 | 0.553 | 0.466 | 0.655 | 0.670 | 0.696 | 0.653 | 0.679 |
| C19 | 0.702 | 0.559 | 0.474 | 0.656 | 0.669 | 0.704 | 0.681 | 0.678 |
| C20 | 0.543 | 0.409 | 0.328 | 0.495 | 0.513 | 0.511 | 0.558 | 0.522 |
| C21 | 0.563 | 0.416 | 0.357 | 0.500 | 0.495 | 0.496 | 0.592 | 0.510 |
| C22 | 0.558 | 0.446 | 0.351 | 0.628 | 0.644 | 0.578 | 0.577 | 0.644 |
| C23 | 0.576 | 0.454 | 0.367 | 0.629 | 0.645 | 0.591 | 0.610 | 0.642 |
| C24 | 0.506 | 0.361 | 0.265 | 0.520 | 0.523 | 0.503 | 0.538 | 0.533 |
| C25 | 0.517 | 0.372 | 0.263 | 0.519 | 0.522 | 0.516 | 0.588 | 0.528 |
| C26 | 0.489 | 0.304 | 0.224 | 0.438 | 0.424 | 0.462 | 0.447 | 0.443 |
| C27 | 0.498 | 0.309 | 0.233 | 0.442 | 0.430 | 0.482 | 0.461 | 0.449 |
| C28 | 0.457 | 0.288 | 0.179 | 0.392 | 0.355 | 0.436 | 0.427 | 0.382 |
| C29 | 0.483 | 0.288 | 0.203 | 0.425 | 0.403 | 0.451 | 0.452 | 0.399 |

## Test AP per context: representative checkpoints

Enhancer-only evaluation; values are for the same representative checkpoint selected above. `ab` adult brain, `e13/e5` embryo stages, `ead/hid/wid` eye-antennal/haltere/wing discs, `lb` larval brain, `o` ovary.

| Ref | ab | e13 | e5 | ead | hid | lb | o | wid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 0.708 | 0.575 | 0.496 | 0.633 | 0.629 | 0.639 | 0.631 | 0.625 |
| C02 | 0.636 | 0.441 | 0.388 | 0.545 | 0.535 | 0.540 | 0.580 | 0.546 |
| C03 | 0.676 | 0.568 | 0.494 | 0.636 | 0.631 | 0.625 | 0.594 | 0.626 |
| C04 | 0.707 | 0.574 | 0.495 | 0.636 | 0.635 | 0.635 | 0.622 | 0.629 |
| C05 | 0.623 | 0.443 | 0.384 | 0.554 | 0.550 | 0.527 | 0.567 | 0.549 |
| C06 | 0.645 | 0.437 | 0.373 | 0.541 | 0.540 | 0.524 | 0.572 | 0.539 |
| C07 | 0.740 | 0.599 | 0.540 | 0.673 | 0.665 | 0.650 | 0.677 | 0.678 |
| C08 | 0.662 | 0.495 | 0.429 | 0.600 | 0.584 | 0.558 | 0.627 | 0.604 |
| C09 | 0.701 | 0.588 | 0.532 | 0.663 | 0.662 | 0.636 | 0.642 | 0.673 |
| C10 | 0.737 | 0.591 | 0.532 | 0.664 | 0.662 | 0.644 | 0.670 | 0.672 |
| C11 | 0.649 | 0.485 | 0.443 | 0.597 | 0.594 | 0.567 | 0.597 | 0.608 |
| C12 | 0.663 | 0.496 | 0.435 | 0.598 | 0.584 | 0.556 | 0.618 | 0.603 |
| C13 | 0.683 | 0.569 | 0.505 | 0.642 | 0.657 | 0.622 | 0.641 | 0.665 |
| C14 | — | — | — | — | — | — | — | — |
| C15 | — | — | — | — | — | — | — | — |
| C16 | — | — | — | — | — | — | — | — |
| C17 | — | — | — | — | — | — | — | — |
| C18 | 0.687 | 0.547 | 0.479 | 0.653 | 0.653 | 0.630 | 0.639 | 0.658 |
| C19 | 0.710 | 0.549 | 0.481 | 0.654 | 0.653 | 0.634 | 0.664 | 0.657 |
| C20 | 0.566 | 0.383 | 0.321 | 0.517 | 0.511 | 0.466 | 0.518 | 0.533 |
| C21 | 0.582 | 0.387 | 0.324 | 0.515 | 0.500 | 0.475 | 0.542 | 0.531 |
| C22 | 0.582 | 0.392 | 0.347 | 0.641 | 0.639 | 0.534 | 0.539 | 0.645 |
| C23 | 0.601 | 0.401 | 0.358 | 0.643 | 0.640 | 0.543 | 0.568 | 0.647 |
| C24 | 0.528 | 0.321 | 0.262 | 0.525 | 0.520 | 0.464 | 0.486 | 0.542 |
| C25 | 0.539 | 0.317 | 0.265 | 0.511 | 0.505 | 0.487 | 0.549 | 0.534 |
| C26 | 0.464 | 0.303 | 0.238 | 0.464 | 0.412 | 0.447 | 0.413 | 0.425 |
| C27 | 0.484 | 0.304 | 0.245 | 0.468 | 0.419 | 0.458 | 0.432 | 0.433 |
| C28 | 0.429 | 0.279 | 0.191 | 0.412 | 0.355 | 0.421 | 0.394 | 0.378 |
| C29 | 0.490 | 0.277 | 0.214 | 0.419 | 0.381 | 0.418 | 0.426 | 0.406 |

## Regressors

One selected checkpoint per architecture. Loss: AlphaGenome-style count/position loss plus a cross-context auxiliary term; training-only track scaling. All ran 40 epochs. The shared LR schedule warms to 1e-4 in one epoch, decays to 5e-5 over four epochs, then uses validation-plateau reductions down to 1e-6. Selection maximizes the scientific composite. See the [scientific contract](reproduction_science.md).

| ID | Input bp | ATAC / H3K27ac target bp | Epoch | Val composite | Val overcorrelation |
| --- | --- | --- | --- | --- | --- |
| `attention_regression` | 2048 | 512 / 1536 | 27 | 0.717 | 0.127 |
| `cnn_regression` | 2048 | 512 / 1536 | 39 | 0.723 | 0.140 |
| `dense_regression` | 2048 | 512 / 1536 | 38 | 0.687 | 0.172 |
| `enhancernet_cnn_regression` | 2048 | 512 / 1536 | 40 | 0.455 | 0.297 |
| `enhancernet_attention_regression` | 2048 | 512 / 1536 | 40 | 0.612 | 0.240 |
| `cnn_1024_regression` | 1024 | 512 / 512 | 39 | 0.680 | 0.166 |
| `cnn_512_regression` | 512 | 512 / 512 | 40 | 0.599 | 0.185 |

### ATAC validation

| Model | All bins r | All bins ρ | All bins R² | Center r | Center ρ | Center R² | Context-pattern r | Breadth ρ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN + attention | 0.818 | 0.767 | 0.653 | 0.827 | 0.771 | 0.671 | 0.654 | 0.788 |
| Dilated CNN | 0.829 | 0.774 | 0.676 | 0.840 | 0.779 | 0.698 | 0.661 | 0.801 |
| Flatten/dense CNN | 0.798 | 0.749 | 0.615 | 0.809 | 0.749 | 0.637 | 0.598 | 0.767 |
| EnhancerNet CNN | 0.470 | 0.407 | 0.052 | 0.547 | 0.493 | 0.186 | 0.318 | 0.463 |
| EnhancerNet + attention | 0.745 | 0.690 | 0.542 | 0.753 | 0.681 | 0.559 | 0.482 | 0.718 |
| Dilated CNN 1,024 | 0.815 | 0.757 | 0.654 | 0.823 | 0.753 | 0.670 | 0.624 | 0.778 |
| Dilated CNN 512 | 0.756 | 0.701 | 0.551 | 0.766 | 0.688 | 0.573 | 0.545 | 0.705 |

### H3K27AC validation

| Model | All bins r | All bins ρ | All bins R² | Center r | Center ρ | Center R² | Context-pattern r | Breadth ρ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN + attention | 0.710 | 0.615 | 0.487 | 0.749 | 0.661 | 0.548 | 0.629 | 0.733 |
| Dilated CNN | 0.720 | 0.615 | 0.508 | 0.766 | 0.670 | 0.580 | 0.624 | 0.741 |
| Flatten/dense CNN | 0.684 | 0.583 | 0.445 | 0.730 | 0.633 | 0.516 | 0.603 | 0.713 |
| EnhancerNet CNN | 0.494 | 0.432 | 0.126 | 0.551 | 0.489 | 0.206 | 0.367 | 0.588 |
| EnhancerNet + attention | 0.622 | 0.511 | 0.370 | 0.683 | 0.574 | 0.455 | 0.521 | 0.687 |
| Dilated CNN 1,024 | 0.687 | 0.576 | 0.461 | 0.711 | 0.605 | 0.496 | 0.563 | 0.638 |
| Dilated CNN 512 | 0.599 | 0.491 | 0.342 | 0.621 | 0.513 | 0.372 | 0.465 | 0.542 |

All-bin metrics pool individual bins within each context and then macro-average the eight context metrics. Center metrics use the central-512-bp mean. Both are measured after the original log1p transform; R² is not squared Pearson correlation. The snapshot retains additional per-context and breadth metrics. Compare matched target spans: 2048-bp models predict wider H3K27ac profiles than the shorter-input regressors.

## Complete checkpoint index

Paths below are relative to the **historical artifact root**, not this source checkout. They identify existing research artifacts but are not download links. Full hashes, last-checkpoint paths and initialization checkpoint hashes are in the JSON snapshot. Weights need a separate versioned deposit before this becomes a public model zoo.

<details>
<summary>All 94 selected checkpoints and per-run validation results</summary>

| Model ID | Type / training | Epoch | Val E AP / composite | Checkpoint SHA256 | Artifact path |
| --- | --- | --- | --- | --- | --- |
| `attention_regression` | regressor / scratch | 27 | 0.717 | `ca6f8a9303af2f45bbdfeacbc1e8149cf0df66708040bf4b6781ff4fe9428704` | `runs/v4_4x_agloss_cecar_20260914/results/v4_4x_agloss_4090_20260912/model/best_model.pt` |
| `attention_scratch_20260914` | classifier / scratch | 38 | 0.535 | `a73474213b0a5b538e41a6dd75b3e882e28d3cfc2d7d3e9b2ef868c90650be6c` | `experiments/classifier_transfer_20260914/runs/attention_scratch_20260914/best_model.pt` |
| `attention_scratch_20260915` | classifier / scratch | 40 | 0.541 | `b96dc258015330baade367708e950ff4dac6fb905f27a4618585b0b5ff54c9be` | `experiments/classifier_transfer_20260914/runs/attention_scratch_20260915/best_model.pt` |
| `attention_scratch_20260916` | classifier / scratch | 37 | 0.535 | `e19e3d4ac32e2bba74a9bd875c7faf3421244a1d8702feb4414f8d8fcdd6682e` | `experiments/classifier_transfer_20260914/runs/attention_scratch_20260916/best_model.pt` |
| `attention_finetune_20260914` | classifier / finetune | 13 | 0.630 | `b093fa84f43306b6fd113c17d45045072bf2e270cba32edb377dc714bf63572e` | `experiments/classifier_transfer_20260914/runs/attention_finetune_20260914/best_model.pt` |
| `attention_finetune_20260915` | classifier / finetune | 12 | 0.630 | `2dd57d8d412f97c59ced725a876931ad4d45924f68aeedc9303773de57f472a0` | `experiments/classifier_transfer_20260914/runs/attention_finetune_20260915/best_model.pt` |
| `attention_finetune_20260916` | classifier / finetune | 11 | 0.629 | `37fe4308158abd9cdf8393935353587d455f4db8ae2ee607fb9d6957cd2f7e5c` | `experiments/classifier_transfer_20260914/runs/attention_finetune_20260916/best_model.pt` |
| `cnn_regression` | regressor / scratch | 39 | 0.723 | `4ae1ea214a8735d264e2c564e95ec89d779c96ccbc7d431f9fb39f1e6717ec53` | `experiments/classifier_transfer_20260914/cnn_regression/model/best_model.pt` |
| `cnn_scratch_20260914` | classifier / scratch | 35 | 0.573 | `980c0f2b14f031b9d1aa2cc0d8284416a62fb8ac0fe1b55b755d4bc99d9eecf9` | `experiments/classifier_transfer_20260914/runs/cnn_scratch_20260914/best_model.pt` |
| `cnn_scratch_20260915` | classifier / scratch | 37 | 0.574 | `661aae63cd3e1cd7a0abe8fb496a70e7979f7a2b818488796f8edfc3d5e5c14c` | `experiments/classifier_transfer_20260914/runs/cnn_scratch_20260915/best_model.pt` |
| `cnn_scratch_20260916` | classifier / scratch | 36 | 0.582 | `0fb14e15fc8502c123e5d5f4ec63caeeb2a02b8c6dcae19de70b4954ed52e3b5` | `experiments/classifier_transfer_20260914/runs/cnn_scratch_20260916/best_model.pt` |
| `cnn_finetune_20260914` | classifier / finetune | 38 | 0.667 | `c6dddf28f8025788ae8dd5348c5607baec7d302a719a71e9d606b86551482d48` | `experiments/classifier_transfer_20260914/runs/cnn_finetune_20260914/best_model.pt` |
| `cnn_finetune_20260915` | classifier / finetune | 37 | 0.666 | `a9cea06291b2f1ba84437ce793f33b5202c189f792be3a3a668244dbf083d275` | `experiments/classifier_transfer_20260914/runs/cnn_finetune_20260915/best_model.pt` |
| `cnn_finetune_20260916` | classifier / finetune | 36 | 0.666 | `075c2e42a97375de1a92d534539c87d503c4dbdda2d771570493e8270f9ecaea` | `experiments/classifier_transfer_20260914/runs/cnn_finetune_20260916/best_model.pt` |
| `dense_regression` | regressor / scratch | 38 | 0.687 | `140e06ec6123cc936d4357edbaf44ca099ac48081fff385dfcf6e927a99990e6` | `experiments/classifier_dense_20260915/dense_regression/model/best_model.pt` |
| `dense_scratch_20260914` | classifier / scratch | 9 | 0.481 | `d59131ab887b57c617c2cce93086eaec92e5b4d590e2ddd261c380ae1a33ad92` | `experiments/classifier_dense_20260915/runs/dense_scratch_20260914/best_model.pt` |
| `dense_scratch_20260915` | classifier / scratch | 9 | 0.491 | `a5ac744ae5a9282f64e4e133e95f6b27c301850da0395ac52a0441f065d07637` | `experiments/classifier_dense_20260915/runs/dense_scratch_20260915/best_model.pt` |
| `dense_scratch_20260916` | classifier / scratch | 9 | 0.485 | `4d7fb4f1048cfada55b03beed2effa5300864ce24564f83872f51b1bdb87c97e` | `experiments/classifier_dense_20260915/runs/dense_scratch_20260916/best_model.pt` |
| `dense_finetune_20260914` | classifier / finetune | 40 | 0.639 | `a14bdd912b53036b169eefea39fdd8ab259f881824042d70b94de3e927409605` | `experiments/classifier_dense_20260915/runs/dense_finetune_20260914/best_model.pt` |
| `dense_finetune_20260915` | classifier / finetune | 39 | 0.639 | `589fe3096cc041f558130e379f663ed9b5c7dfb2117cebe9037adad0dd9b0624` | `experiments/classifier_dense_20260915/runs/dense_finetune_20260915/best_model.pt` |
| `dense_finetune_20260916` | classifier / finetune | 36 | 0.640 | `12b761323db412664b87179536ee94a950b16a30d415d3be6e2f17a66e5a8cba` | `experiments/classifier_dense_20260915/runs/dense_finetune_20260916/best_model.pt` |
| `enhancernet_cnn_regression` | regressor / scratch | 40 | 0.455 | `625652039a55082136f8e038d41155223ce1bb9281ba06522f3f40f46547093a` | `experiments/classifier_enhancernet_20260915/enhancernet_cnn_regression/model/best_model.pt` |
| `enhancernet_cnn_scratch_20260914` | classifier / scratch | 40 | 0.379 | `5821c53d7f61fcee6a5c77c8d1d1a11984c69da04a9de9b300c817be3267dda5` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_cnn_scratch_20260914/best_model.pt` |
| `enhancernet_cnn_scratch_20260915` | classifier / scratch | 40 | 0.388 | `5ebb6cabfdd9a2df0b67df7bd042ed7589e69b95d8c28955d653c4fb81c87ced` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_cnn_scratch_20260915/best_model.pt` |
| `enhancernet_cnn_scratch_20260916` | classifier / scratch | 40 | 0.387 | `d196580afe36366a0d40dfb358a0f59b35f3c03610f7eb3c88c0c32126e591ae` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_cnn_scratch_20260916/best_model.pt` |
| `enhancernet_cnn_finetune_20260914` | classifier / finetune | 40 | 0.412 | `16463b91ff4d7515d16254a4950746e82c665679f596baa9bef6dc45b5e7adad` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_cnn_finetune_20260914/best_model.pt` |
| `enhancernet_cnn_finetune_20260915` | classifier / finetune | 39 | 0.413 | `61d8dff95e917507bf369a777e52fde450b7a5460b5f55c98f492671f46f2c3c` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_cnn_finetune_20260915/best_model.pt` |
| `enhancernet_cnn_finetune_20260916` | classifier / finetune | 38 | 0.413 | `ca39a62a0f9cf979354ac79dcb55df45283bad0bf40f5ec71aaa5f9252cc79e4` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_cnn_finetune_20260916/best_model.pt` |
| `enhancernet_attention_regression` | regressor / scratch | 40 | 0.612 | `d239c76b478cb9f1ec25702eb69dc893adffd1d49f7abde9f4e2815638df7dc8` | `experiments/classifier_enhancernet_20260915/enhancernet_attention_regression/model/best_model.pt` |
| `enhancernet_attention_scratch_20260914` | classifier / scratch | 40 | 0.475 | `f852db8ece5244f5fac5c100b1bcd0ef8274441810cb5a8259c81b037449a8f1` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_attention_scratch_20260914/best_model.pt` |
| `enhancernet_attention_scratch_20260915` | classifier / scratch | 40 | 0.478 | `a9cf8e8302ec7aac6f86b246b2b104e119ef23b93902131213d2a764e1b0e2c2` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_attention_scratch_20260915/best_model.pt` |
| `enhancernet_attention_scratch_20260916` | classifier / scratch | 39 | 0.470 | `30c6df8a7f82678ceea7394d4226abd45c556ad6bdfd3251f5a789bad46b24cc` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_attention_scratch_20260916/best_model.pt` |
| `enhancernet_attention_finetune_20260914` | classifier / finetune | 39 | 0.564 | `16f53b2ef07e65453b01161bb9f6637f295a9414de66f02fb409bac131a9623d` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_attention_finetune_20260914/best_model.pt` |
| `enhancernet_attention_finetune_20260915` | classifier / finetune | 40 | 0.564 | `f4a46f64ebc0cc6acf0af95311e3478eca09691c61889ade96fe9a66e0b3d2fc` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_attention_finetune_20260915/best_model.pt` |
| `enhancernet_attention_finetune_20260916` | classifier / finetune | 37 | 0.563 | `ba481ca81e97d805d567a2e22fe73fb75d790ea9a42095b6846473993be5d616` | `experiments/classifier_enhancernet_20260915/runs/enhancernet_attention_finetune_20260916/best_model.pt` |
| `retained_heads__attention_scratch_20260914` | classifier / scratch | 38 | 0.539 | `b83a395861fa33bbd5c57b0d451047d5a6d9de09a762a095e01f8a63dd530ba3` | `experiments/classifier_retained_heads_20260915/runs/attention_scratch_20260914/best_model.pt` |
| `retained_heads__attention_scratch_20260915` | classifier / scratch | 32 | 0.538 | `b4294e873ab6891f9496995ce1286c2ad6cd39f977f56dd22ee267f129ebb231` | `experiments/classifier_retained_heads_20260915/runs/attention_scratch_20260915/best_model.pt` |
| `retained_heads__attention_scratch_20260916` | classifier / scratch | 39 | 0.540 | `dbe7f0ebb83260fb843b2198fb94101a69cdc2e244b3e852748fa7fe434bba4c` | `experiments/classifier_retained_heads_20260915/runs/attention_scratch_20260916/best_model.pt` |
| `retained_heads__attention_finetune_20260914` | classifier / finetune | 13 | 0.629 | `5990347d89fed9df826772d33f6f38f9363e3c86a1b89be1f96fe08fcb9bd919` | `experiments/classifier_retained_heads_20260915/runs/attention_finetune_20260914/best_model.pt` |
| `retained_heads__attention_finetune_20260915` | classifier / finetune | 18 | 0.628 | `b75834c25ffe3dd95d29c7f584cee5e39f26b23f6b6cc60495f40d957a6a41a4` | `experiments/classifier_retained_heads_20260915/runs/attention_finetune_20260915/best_model.pt` |
| `retained_heads__attention_finetune_20260916` | classifier / finetune | 15 | 0.629 | `93ad6c4fc4061c4f69a2d9ec008e80767d93a7af4a0456bf2241129af8b44b51` | `experiments/classifier_retained_heads_20260915/runs/attention_finetune_20260916/best_model.pt` |
| `retained_heads__cnn_scratch_20260914` | classifier / scratch | 38 | 0.576 | `d0b367645ea016abc742f9b83fe7f706c09b39ec597755df9643bcd3b3412a1c` | `experiments/classifier_retained_heads_20260915/runs/cnn_scratch_20260914/best_model.pt` |
| `retained_heads__cnn_scratch_20260915` | classifier / scratch | 37 | 0.580 | `550245e4fc4635ef6e325e8f0751319795135fc195e8a6b799452c4eaf71a35d` | `experiments/classifier_retained_heads_20260915/runs/cnn_scratch_20260915/best_model.pt` |
| `retained_heads__cnn_scratch_20260916` | classifier / scratch | 37 | 0.585 | `31719df514cc664f32246ccd7918051ccbd918e34b6fa6f6ca5b9435bd4a7b9f` | `experiments/classifier_retained_heads_20260915/runs/cnn_scratch_20260916/best_model.pt` |
| `retained_heads__cnn_finetune_20260914` | classifier / finetune | 38 | 0.658 | `c00e680b24371a275e5de4e4957308ad63be1297fbf592681dcd55c55d969c86` | `experiments/classifier_retained_heads_20260915/runs/cnn_finetune_20260914/best_model.pt` |
| `retained_heads__cnn_finetune_20260915` | classifier / finetune | 37 | 0.656 | `7e822ef1989061e51e8f7bf9196928ca42b67b504321ce726620cff6dce74eeb` | `experiments/classifier_retained_heads_20260915/runs/cnn_finetune_20260915/best_model.pt` |
| `retained_heads__cnn_finetune_20260916` | classifier / finetune | 36 | 0.657 | `675ea473bc1657a4b525f66ed1c718da7a3dee20288be88de4eecbb390c1aac8` | `experiments/classifier_retained_heads_20260915/runs/cnn_finetune_20260916/best_model.pt` |
| `background__attention_scratch_20260914` | classifier / scratch | 40 | 0.534 | `fc31e67910f3f251ff22b142265ef31327f492ad52db0081aacacdef51827c2c` | `experiments/classifier_background_20260916/runs/attention_scratch_20260914/best_model.pt` |
| `background__attention_scratch_20260915` | classifier / scratch | 38 | 0.535 | `8472f2e7e413345b31e8e2da03476a1bf8e3dbb1af3aef053a948e734292e9b6` | `experiments/classifier_background_20260916/runs/attention_scratch_20260915/best_model.pt` |
| `background__attention_scratch_20260916` | classifier / scratch | 40 | 0.533 | `89a624a0c55121db97933954840072f775dc576e5c05a8e70001862ac51de5d3` | `experiments/classifier_background_20260916/runs/attention_scratch_20260916/best_model.pt` |
| `background__attention_finetune_20260914` | classifier / finetune | 26 | 0.620 | `dc8ec962b2a6e4f0c7191f75286dfe975f1b483ab9c0505629776e49d128f6b8` | `experiments/classifier_background_20260916/runs/attention_finetune_20260914/best_model.pt` |
| `background__attention_finetune_20260915` | classifier / finetune | 12 | 0.619 | `db17bb23e961d8b3a9fdf7c997e4c2a5b6ce7b28af610331c7731ad7ce5795bc` | `experiments/classifier_background_20260916/runs/attention_finetune_20260915/best_model.pt` |
| `background__attention_finetune_20260916` | classifier / finetune | 10 | 0.621 | `a721bc1b5a95441ba26b0e85177827303afa18c470d765608eb50cb03a83a2ec` | `experiments/classifier_background_20260916/runs/attention_finetune_20260916/best_model.pt` |
| `background__cnn_scratch_20260914` | classifier / scratch | 32 | 0.571 | `c0162ce79c04a4b3d0dc22d7823c0d69a874484df065e865e4a0223ad70131db` | `experiments/classifier_background_20260916/runs/cnn_scratch_20260914/best_model.pt` |
| `background__cnn_scratch_20260915` | classifier / scratch | 36 | 0.579 | `da525b1300255bcf3461ad5250257690531111e5a54e95ceff8d88b70db77ccc` | `experiments/classifier_background_20260916/runs/cnn_scratch_20260915/best_model.pt` |
| `background__cnn_scratch_20260916` | classifier / scratch | 40 | 0.582 | `a52cb9795eb2c86ca6f73e70e8767c8f6bfacf161c9f1340b2a1a1a3d90d3727` | `experiments/classifier_background_20260916/runs/cnn_scratch_20260916/best_model.pt` |
| `background__cnn_finetune_20260914` | classifier / finetune | 38 | 0.649 | `7dc8eb721fbd4f292eba95ff2ea44399cfd1aae5a596e8d56e6d7ff25a95628a` | `experiments/classifier_background_20260916/runs/cnn_finetune_20260914/best_model.pt` |
| `background__cnn_finetune_20260915` | classifier / finetune | 37 | 0.647 | `e16e32894c7c1d7f5714d0c7e1bb88a1f4a71b5ccebf0bdb8e9251db5adfb8a0` | `experiments/classifier_background_20260916/runs/cnn_finetune_20260915/best_model.pt` |
| `background__cnn_finetune_20260916` | classifier / finetune | 38 | 0.648 | `d63d6561c976390114f32772d2331720d70f0bb9bc075ee5b3d6745ae553d38c` | `experiments/classifier_background_20260916/runs/cnn_finetune_20260916/best_model.pt` |
| `background__dense_scratch_20260914` | classifier / scratch | 11 | 0.476 | `b60eeb00063680ada46367c709dbb1762abf92ff528eb23f9e68d6437e12c808` | `experiments/classifier_background_20260916/runs/dense_scratch_20260914/best_model.pt` |
| `background__dense_scratch_20260915` | classifier / scratch | 9 | 0.485 | `f09e8061a6fde8ba782f11b2734801f1fc96c3b76809a2cde09c9591e860b9ce` | `experiments/classifier_background_20260916/runs/dense_scratch_20260915/best_model.pt` |
| `background__dense_scratch_20260916` | classifier / scratch | 9 | 0.478 | `3a303383e29af2d7c2e59d5c50682051f11a544d40c43d7fc5e85aa56d08d06f` | `experiments/classifier_background_20260916/runs/dense_scratch_20260916/best_model.pt` |
| `background__dense_finetune_20260914` | classifier / finetune | 40 | 0.630 | `29118fe3c274f2e884cf0342274a3f3a3d4029187356a8f682aea8c4fbed075c` | `experiments/classifier_background_20260916/runs/dense_finetune_20260914/best_model.pt` |
| `background__dense_finetune_20260915` | classifier / finetune | 39 | 0.630 | `c6d1599df558e4504db82c4420bc85b3f838b13cf0c81afd15f71c22d847071c` | `experiments/classifier_background_20260916/runs/dense_finetune_20260915/best_model.pt` |
| `background__dense_finetune_20260916` | classifier / finetune | 36 | 0.632 | `43f0536823c2f7765c988ba9b9362d7c4b88c652e6a3e1c5973db2c30a190c80` | `experiments/classifier_background_20260916/runs/dense_finetune_20260916/best_model.pt` |
| `background__enhancernet_cnn_scratch_20260914` | classifier / scratch | 37 | 0.359 | `f38399d1f487a7b1bcb0789a863098112a7216d469fc2e7a87451223b97dd96c` | `experiments/classifier_background_20260916/runs/enhancernet_cnn_scratch_20260914/best_model.pt` |
| `background__enhancernet_cnn_scratch_20260915` | classifier / scratch | 37 | 0.364 | `0633ccbb0ff3a13b984bf3aaa37846491359b501f37fa7f9b2a80e0413412747` | `experiments/classifier_background_20260916/runs/enhancernet_cnn_scratch_20260915/best_model.pt` |
| `background__enhancernet_cnn_scratch_20260916` | classifier / scratch | 35 | 0.358 | `2e33b5fcb3fec96073aebe5a8d2e6a0b4c3427b3340db8037fabc421a5cf3fbb` | `experiments/classifier_background_20260916/runs/enhancernet_cnn_scratch_20260916/best_model.pt` |
| `background__enhancernet_cnn_finetune_20260914` | classifier / finetune | 40 | 0.404 | `ab80fafbebd65c2a0de22a70029f4d30bb415cf54a544c50add3db31872c9e3e` | `experiments/classifier_background_20260916/runs/enhancernet_cnn_finetune_20260914/best_model.pt` |
| `background__enhancernet_cnn_finetune_20260915` | classifier / finetune | 39 | 0.404 | `1a764f20caac7b7eff5bd1ee300aab394bbba29e8a807bb18af31e6f5f54c35c` | `experiments/classifier_background_20260916/runs/enhancernet_cnn_finetune_20260915/best_model.pt` |
| `background__enhancernet_cnn_finetune_20260916` | classifier / finetune | 38 | 0.404 | `ecac42082f4d27aaaf6634a1e02153fb32fd6a3bec2d8bbbd17f7e866b9377c0` | `experiments/classifier_background_20260916/runs/enhancernet_cnn_finetune_20260916/best_model.pt` |
| `background__enhancernet_attention_scratch_20260914` | classifier / scratch | 38 | 0.460 | `d78615b7ef68851c8eb27365d9f5b36dbef6598d4924b86fe683a9a2c3444deb` | `experiments/classifier_background_20260916/runs/enhancernet_attention_scratch_20260914/best_model.pt` |
| `background__enhancernet_attention_scratch_20260915` | classifier / scratch | 40 | 0.469 | `51fabf2f1eb2a8c34537c34f6cb98969d67ab0e401b3720b95be109b558e626a` | `experiments/classifier_background_20260916/runs/enhancernet_attention_scratch_20260915/best_model.pt` |
| `background__enhancernet_attention_scratch_20260916` | classifier / scratch | 40 | 0.463 | `9bfdeabf265c39109ec3d97ef90305642fdf0d90d87ae5598a6e77b05d440a2c` | `experiments/classifier_background_20260916/runs/enhancernet_attention_scratch_20260916/best_model.pt` |
| `background__enhancernet_attention_finetune_20260914` | classifier / finetune | 39 | 0.553 | `a08e6480103cdc983372b19499f40501d0c233fb52310d46007fe5edf5bc0dc5` | `experiments/classifier_background_20260916/runs/enhancernet_attention_finetune_20260914/best_model.pt` |
| `background__enhancernet_attention_finetune_20260915` | classifier / finetune | 38 | 0.553 | `0149fe1830a29721bffc879f1bbd86e78b37f9634dd8af83d03ec7319156ed74` | `experiments/classifier_background_20260916/runs/enhancernet_attention_finetune_20260915/best_model.pt` |
| `background__enhancernet_attention_finetune_20260916` | classifier / finetune | 40 | 0.554 | `7f4a5edb97d54dff80ade5de1966e4405f40702b3688759bda316b68779d42cf` | `experiments/classifier_background_20260916/runs/enhancernet_attention_finetune_20260916/best_model.pt` |
| `cnn_1024_regression` | regressor / scratch | 39 | 0.680 | `cbb10a4b19c314de3141402aa897fb7103b284afd05bf2e73d516c638752000a` | `experiments/short_context_20260916/cnn_1024_regression/model/best_model.pt` |
| `cnn_1024_finetune_20260914` | classifier / finetune | 38 | 0.632 | `a31d31e66b4b984969f261f1c9ead846da126f02f27d883283cf8db52709481b` | `experiments/short_context_20260916/runs/cnn_1024_finetune_20260914/best_model.pt` |
| `cnn_1024_finetune_20260915` | classifier / finetune | 37 | 0.632 | `94eb88e0160aac14745fcdb907f799fceac4682b59d6deb9b0ab90803b03c8fd` | `experiments/short_context_20260916/runs/cnn_1024_finetune_20260915/best_model.pt` |
| `cnn_1024_finetune_20260916` | classifier / finetune | 37 | 0.633 | `0c921ab8324061443a6f482963f1e17bf5caa2ee84e3e797af5bedb33792b37d` | `experiments/short_context_20260916/runs/cnn_1024_finetune_20260916/best_model.pt` |
| `cnn_512_regression` | regressor / scratch | 40 | 0.599 | `cb1156ff4abdc90ebe50ce0bf7b7e533801a17b9adbca1716b0e5cf8290bd2bc` | `experiments/length_transfer_20260918/revision_v2/cnn_512_regression/model/best_model.pt` |
| `cnn_512_scratch_20260914` | classifier / scratch | 15 | 0.522 | `d2bcdeb99f88e2707d50cc540329f676ca9f72119526bbea81e8a10eaf12e29c` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_scratch_20260914/best_model.pt` |
| `cnn_512_scratch_20260915` | classifier / scratch | 15 | 0.525 | `14b14b032c1e00d342526c64c7a43aa75f1df9fe4a1772de156a7975ed0bc38b` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_scratch_20260915/best_model.pt` |
| `cnn_512_scratch_20260916` | classifier / scratch | 15 | 0.521 | `5a4d6f29e21c681341c290fc346bb54b5f12d3c562eb433447196e0fc52b63a6` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_scratch_20260916/best_model.pt` |
| `cnn_512_from1024_20260914` | classifier / finetune | 15 | 0.615 | `eaa29e88c741f70159e8cbe122bede689d21d3a6056f1c1d6da66bf5e067fbe0` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_from1024_20260914/best_model.pt` |
| `cnn_512_from1024_20260915` | classifier / finetune | 15 | 0.616 | `491df32ce5ca2c1441b0b9dd2eb8087a82907e3e717badee9ef933232eedfa3c` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_from1024_20260915/best_model.pt` |
| `cnn_512_from1024_20260916` | classifier / finetune | 15 | 0.616 | `adda7c53b97769b63a7078558010191369312d5c12381d2a4c240ef9e602ab90` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_from1024_20260916/best_model.pt` |
| `cnn_512_from2048_20260914` | classifier / finetune | 15 | 0.618 | `92afa6b65cbb00d73242bf4d96a80d7dc75161b4769211169323e933db65aede` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_from2048_20260914/best_model.pt` |
| `cnn_512_from2048_20260915` | classifier / finetune | 15 | 0.616 | `b26b66c30fa7d6a8a299fd5fed33f0456057cc1e4128dd55e1ad3025e5570904` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_from2048_20260915/best_model.pt` |
| `cnn_512_from2048_20260916` | classifier / finetune | 15 | 0.616 | `0ce51342468f9dc9eb3f6a721100a5c1f4b60535a37417568bbb6855c13f1e6e` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_from2048_20260916/best_model.pt` |
| `cnn_512_from512_20260914` | classifier / finetune | 15 | 0.601 | `12974ed7272f89e1709aa33006c8a7dc2e8b9c2079fc1c2a3f26e284542f1a65` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_from512_20260914/best_model.pt` |
| `cnn_512_from512_20260915` | classifier / finetune | 15 | 0.602 | `34b8d5d14c2a9909113512e83b8dc441e085669794514fe8a3f807d58a6c6f2f` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_from512_20260915/best_model.pt` |
| `cnn_512_from512_20260916` | classifier / finetune | 15 | 0.602 | `3ab2107c425f3b567cbf54e307a8ebf999615b25bb8868952238a1aa7744145c` | `experiments/length_transfer_20260918/revision_v2/runs/cnn_512_from512_20260916/best_model.pt` |

</details>

## Rebuild and provenance

```bash
python scripts/summarize_paper_benchmarks.py --check
# After an explicitly reviewed snapshot update:
python scripts/summarize_paper_benchmarks.py --write
```

This is CPU-only report generation from saved metrics. It does not evaluate models or change scientific results. New portable runs have their own `model_inventory.tsv`; they do not overwrite this historical benchmark.

- Source inventory: `results/model_inventory_20260919/model_inventory.tsv`.
- Source inventory SHA256: `a40197312967b1122bcedaa2971508b5ddf7feb87da64de5acd38046fad67ef5`.
- Original source-snapshot SHA256: `3729762b66ff06d7cab8fd51b6178ff504ddb10a02e51e7eed2a28f387b42242`.
- Snapshot contains no home-directory paths, login names, credentials, sequences or weights.
