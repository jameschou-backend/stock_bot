# Debug Evaluation Model shadow_monitor_model_1m_2026-01-02_2026-02-02

- invalid_result: `False`
- periods_checked: `1`
- anomaly_rows: `0`
- inf/nan root cause path: `period_return = (exit_px / entry_px - 1) - tx_cost`；若 `entry_px=0` 會導致除零，進而污染 total_return/cagr/average_holding_return 與波動指標。

## Period Diagnostics
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
