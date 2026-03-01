# Debug Evaluation Model model_strict_20260228

- invalid_result: `False`
- periods_checked: `6`
- anomaly_rows: `24`
- inf/nan root cause path: `period_return = (exit_px / entry_px - 1) - tx_cost`；若 `entry_px=0` 會導致除零，進而污染 total_return/cagr/average_holding_return 與波動指標。

## Period Diagnostics
### 2025-04-01 -> 2025-05-02
- period_return: `-0.041870475298869655`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 2314 | 12.2 | 11.15 | False | False | False |
| 2380 | 3.99 | 3.88 | False | False | False |
| 2911 | 5.11 | 5.22 | False | False | False |
| 3081 | 287.0 | 278.0 | False | False | False |
| 3128 | 23.55 | 26.45 | False | False | False |
| 3363 | 224.5 | 191.0 | False | False | False |
| 3447 | 56.8 | 51.2 | False | False | False |
| 4770 | 274.0 | 248.0 | False | False | False |
| 5225 | 108.0 | 105.0 | False | False | False |
| 5314 | 79.4 | 64.4 | False | False | False |
| 6125 | 68.0 | 63.9 | False | False | False |
| 6716 | 61.0 | 54.5 | False | False | False |
| 6820 | 77.7 | 82.5 | False | False | False |
| 6854 | 145.5 | 147.0 | False | False | False |
| 7530 | 16.2 | 14.65 | False | False | False |
| 7781 | 4.98 | 4.82 | False | False | False |
| 7796 | 151.0 | 148.5 | False | False | False |
| 8941 | 0.0 | 53.0 | True | False | False |
| 9103 | 4.8 | 5.14 | False | False | False |

### 2025-07-01 -> 2025-08-01
- period_return: `0.2209567728273473`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 1773 | 122.0 | 126.5 | False | False | False |
| 2380 | 3.99 | 4.11 | False | False | False |
| 2867 | 5.0 | 5.21 | False | False | False |
| 2911 | 5.44 | 4.97 | False | False | False |
| 3018 | 16.2 | 13.4 | False | False | False |
| 3095 | 0.0 | 28.8 | True | False | False |
| 3259 | 0.0 | 21.65 | True | False | False |
| 3516 | 0.0 | 20.05 | True | False | False |
| 4117 | 16.1 | 15.9 | False | False | False |
| 4194 | 6.19 | 10.6 | False | False | False |
| 4414 | 3.28 | 11.75 | False | False | False |
| 4763 | 90.8 | 78.7 | False | False | False |
| 4804 | 6.2 | 9.76 | False | False | False |
| 5701 | 4.9 | 4.74 | False | False | False |
| 6434 | 2.4 | 2.32 | False | False | False |
| 7731 | 5.1 | 4.89 | False | False | False |
| 7737 | 0.0 | 49.0 | True | False | False |
| 7781 | 5.07 | 5.27 | False | False | False |
| 8933 | 6.02 | 6.41 | False | False | False |
| 9103 | 5.02 | 4.96 | False | False | False |

### 2025-08-01 -> 2025-09-01
- period_return: `0.11330613006813141`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 1343 | 48.75 | 49.5 | False | False | False |
| 1436 | 72.9 | 70.1 | False | False | False |
| 2380 | 4.11 | 6.08 | False | False | False |
| 2867 | 5.21 | 6.11 | False | False | False |
| 3310 | 64.7 | 63.5 | False | False | False |
| 3321 | 7.25 | 7.45 | False | False | False |
| 3593 | 6.61 | 7.25 | False | False | False |
| 4529 | 3.78 | 3.79 | False | False | False |
| 4738 | 5.0 | 6.05 | False | False | False |
| 4804 | 9.76 | 10.0 | False | False | False |
| 5355 | 10.4 | 9.44 | False | False | False |
| 5488 | 9.58 | 10.5 | False | False | False |
| 5508 | 80.2 | 71.3 | False | False | False |
| 6189 | 59.0 | 50.8 | False | False | False |
| 6403 | 6.28 | 7.69 | False | False | False |
| 6652 | 5.92 | 7.28 | False | False | False |
| 6673 | 6.02 | 7.3 | False | False | False |
| 6881 | 0.0 | 224.0 | True | False | False |
| 6919 | 121.0 | 210.5 | False | False | False |
| 7781 | 5.27 | 5.52 | False | False | False |

### 2025-09-01 -> 2025-10-01
- period_return: `0.09243364088230423`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 2235 | 0.0 | 0.0 | True | False | False |
| 2327 | 135.0 | 170.0 | False | False | False |
| 2364 | 74.3 | 74.0 | False | False | False |
| 2539 | 48.05 | 48.35 | False | False | False |
| 2726 | 0.0 | 16.05 | True | False | False |
| 2867 | 6.11 | 6.11 | False | False | False |
| 2911 | 4.8 | 5.1 | False | False | False |
| 3054 | 0.0 | 24.7 | True | False | False |
| 3056 | 20.5 | 19.9 | False | False | False |
| 3117 | 5.2 | 5.6 | False | False | False |
| 4166 | 23.0 | 26.3 | False | False | False |
| 4303 | 66.2 | 74.1 | False | False | False |
| 4529 | 3.79 | 3.78 | False | False | False |
| 4735 | 0.0 | 34.0 | True | False | False |
| 5701 | 4.95 | 4.55 | False | False | False |
| 6434 | 3.29 | 5.79 | False | False | False |
| 6518 | 0.0 | 26.95 | True | False | False |
| 6847 | 0.0 | 74.7 | True | False | False |
| 7731 | 4.41 | 4.01 | False | False | False |
| 8923 | 0.0 | 22.55 | True | False | False |

### 2025-10-01 -> 2025-11-03
- period_return: `0.006051309548425749`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 1570 | 0.0 | 30.5 | True | False | False |
| 2035 | 0.0 | 28.3 | True | False | False |
| 2245 | 37.15 | 28.95 | False | False | False |
| 2540 | 56.0 | 50.4 | False | False | False |
| 2743 | 89.4 | 89.0 | False | False | False |
| 2867 | 6.11 | 7.21 | False | False | False |
| 2924 | 22.1 | 0.0 | True | False | False |
| 3049 | 6.77 | 6.2 | False | False | False |
| 3260 | 0.0 | 194.5 | True | False | False |
| 3603 | 0.0 | 5.84 | True | False | False |
| 4529 | 3.78 | 3.76 | False | False | False |
| 4585 | 426.5 | 407.0 | False | False | False |
| 4976 | 37.45 | 34.4 | False | False | False |
| 4991 | 135.0 | 175.0 | False | False | False |
| 5481 | 20.35 | 19.95 | False | False | False |
| 6434 | 5.79 | 5.04 | False | False | False |
| 6793 | 6.05 | 5.3 | False | False | False |
| 6932 | 5.01 | 6.72 | False | False | False |
| 7607 | 13.55 | 15.0 | False | False | False |
| 7731 | 4.01 | 4.02 | False | False | False |

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
