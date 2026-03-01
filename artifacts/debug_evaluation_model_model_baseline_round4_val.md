# Debug Evaluation Model model_baseline_round4_val

- invalid_result: `False`
- periods_checked: `3`
- anomaly_rows: `9`
- inf/nan root cause path: `period_return = (exit_px / entry_px - 1) - tx_cost`；若 `entry_px=0` 會導致除零，進而污染 total_return/cagr/average_holding_return 與波動指標。

## Period Diagnostics
### 2025-11-03 -> 2025-12-01
- period_return: `-0.009132057233514504`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 1340 | 6.2 | 7.09 | False | False | False |
| 1463 | 17.0 | 19.3 | False | False | False |
| 2429 | 94.5 | 80.0 | False | False | False |
| 2607 | 57.6 | 58.2 | False | False | False |
| 3049 | 6.2 | 6.82 | False | False | False |
| 4529 | 3.76 | 3.88 | False | False | False |
| 4738 | 6.3 | 6.35 | False | False | False |
| 5701 | 4.0 | 4.0 | False | False | False |
| 6221 | 37.1 | 32.7 | False | False | False |
| 6259 | 0.0 | 15.95 | True | False | False |
| 6434 | 5.04 | 4.39 | False | False | False |
| 6535 | 211.5 | 202.0 | False | False | False |
| 6622 | 0.0 | 7.9 | True | False | False |
| 6682 | 33.85 | 28.25 | False | False | False |
| 6839 | 13.15 | 12.4 | False | False | False |
| 6946 | 12.5 | 10.75 | False | False | False |
| 7731 | 4.02 | 5.17 | False | False | False |
| 7781 | 5.29 | 5.09 | False | False | False |
| 9103 | 4.69 | 4.69 | False | False | False |
| 9960 | 0.0 | 20.85 | True | False | False |

### 2025-12-01 -> 2026-01-02
- period_return: `-0.00928477189661515`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 1337 | 4.92 | 4.79 | False | False | False |
| 2380 | 4.63 | 4.62 | False | False | False |
| 2938 | 0.0 | 24.5 | True | False | False |
| 3018 | 0.0 | 11.45 | True | False | False |
| 3494 | 6.96 | 8.67 | False | False | False |
| 3632 | 0.0 | 9.6 | True | False | False |
| 3710 | 5.88 | 5.97 | False | False | False |
| 4529 | 3.88 | 3.78 | False | False | False |
| 5310 | 0.0 | 0.0 | True | False | False |
| 6434 | 4.39 | 4.1 | False | False | False |
| 6610 | 19.45 | 16.7 | False | False | False |
| 6614 | 50.2 | 50.4 | False | False | False |
| 6887 | 0.0 | 60.2 | True | False | False |
| 7575 | 0.0 | 28.8 | True | False | False |
| 7711 | 262.5 | 259.5 | False | False | False |
| 7731 | 5.17 | 4.86 | False | False | False |
| 7781 | 5.09 | 4.66 | False | False | False |
| 7786 | 156.0 | 157.5 | False | False | False |
| 8277 | 8.65 | 9.36 | False | False | False |
| 9103 | 4.69 | 4.46 | False | False | False |

### 2026-01-02 -> 2026-02-02
- period_return: `0.2920968448342121`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 1337 | 4.79 | 5.62 | False | False | False |
| 1417 | 8.92 | 8.65 | False | False | False |
| 1460 | 6.39 | 6.55 | False | False | False |
| 2022 | 8.12 | 8.83 | False | False | False |
| 2380 | 4.62 | 4.94 | False | False | False |
| 2911 | 5.64 | 5.19 | False | False | False |
| 3049 | 6.1 | 10.2 | False | False | False |
| 3117 | 6.8 | 6.94 | False | False | False |
| 4194 | 7.07 | 8.53 | False | False | False |
| 4529 | 3.78 | 3.33 | False | False | False |
| 4738 | 5.64 | 7.09 | False | False | False |
| 5701 | 4.09 | 3.9 | False | False | False |
| 6434 | 4.1 | 5.63 | False | False | False |
| 6682 | 24.25 | 26.2 | False | False | False |
| 6884 | 38.2 | 34.8 | False | False | False |
| 7731 | 4.86 | 6.11 | False | False | False |
| 7781 | 4.66 | 20.4 | False | False | False |
| 8933 | 6.02 | 7.08 | False | False | False |
| 9957 | 6.22 | 7.25 | False | False | False |
