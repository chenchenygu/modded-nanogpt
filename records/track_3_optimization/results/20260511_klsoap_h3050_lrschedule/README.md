# KL-SOAP-H LR Schedule 3050 Results

This result uses KL-SOAP-H from PR #290 with a superlinear learning-rate
cooldown (`LR_POWER=1.5`), nonzero learning-rate floors, and the
`shampoo_beta=0.9` tuple from that submission. It runs for a fixed 3050
training steps while evaluating the learning-rate schedule over a 3125-step
horizon.

The archived script hardcodes the submitted hyperparameter defaults. It reads
`KL_SOAP_SEED` only to select the random seed for a reproducibility run.

## Configuration

| field | value |
|---|---|
| train steps | 3050 |
| schedule steps | 3125 |
| LR power | 1.5 |
| Adam LR floor | 0.03933816945537044 |
| KL-SOAP LR floor | 0.016930477847468664 |
| KL-SOAP lr | 0.018 |
| KL-SOAP beta1 | 0.95 |
| KL-SOAP beta2 | 0.9 |
| KL-SOAP shampoo beta | 0.9 |
| KL-SOAP precondition frequency | 1 |

## Reliability

Across 11 non-cherry-picked seeds at the fixed stopping point K=3050, the mean
validation loss is 3.27873909. The Track 3 significance rule is:

`(3.28 - mean) * sqrt(n) >= 0.004`

For these runs:

`(3.28 - 3.27873909) * sqrt(11) = 0.00418196 >= 0.004`

| seed | val loss at K=3050 | logfile |
|---:|---:|---|
| 0 | 3.27806 | `seed00.txt` |
| 1 | 3.28110 | `seed01.txt` |
| 2 | 3.27866 | `seed02.txt` |
| 3 | 3.27724 | `seed03.txt` |
| 4 | 3.27834 | `seed04.txt` |
| 5 | 3.27795 | `seed05.txt` |
| 6 | 3.27896 | `seed06.txt` |
| 7 | 3.28001 | `seed07.txt` |
| 8 | 3.27837 | `seed08.txt` |
| 9 | 3.27867 | `seed09.txt` |
| 10 | 3.27877 | `seed10.txt` |
| **Mean** | **3.27873909** | |

All logs report `step:3050/3050`; no per-run early stopping is used.

## Reproduction

Run one seed from the repo root with the Track 3 quickstart environment:

```bash
KL_SOAP_SEED=0 torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) records/track_3_optimization/results/20260511_klsoap_h3050_lrschedule/train_gpt_simple_klsoap_h3050_lrschedule.py
```
