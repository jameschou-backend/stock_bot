# Debug Evaluation Model regime_model_round5

- invalid_result: `False`
- periods_checked: `11`
- anomaly_rows: `39`
- inf/nan root cause path: `period_return = (exit_px / entry_px - 1) - tx_cost`；若 `entry_px=0` 會導致除零，進而污染 total_return/cagr/average_holding_return 與波動指標。

## Period Diagnostics
### 2025-03-03 -> 2025-04-01
- period_return: `-0.09662975170917766`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 1231 | 130.5 | 129.0 | False | False | False |
| 1512 | 7.1 | 6.98 | False | False | False |
| 2380 | 4.89 | 3.99 | False | False | False |
| 2867 | 6.93 | 6.23 | False | False | False |
| 3081 | 352.5 | 287.0 | False | False | False |
| 3321 | 11.45 | 7.11 | False | False | False |
| 3593 | 7.61 | 8.3 | False | False | False |
| 4414 | 3.85 | 3.7 | False | False | False |
| 4587 | 0.0 | 32.85 | True | False | False |
| 4991 | 114.5 | 111.0 | False | False | False |
| 5701 | 5.07 | 5.3 | False | False | False |
| 6434 | 3.04 | 2.82 | False | False | False |
| 6451 | 183.5 | 162.5 | False | False | False |
| 6734 | 6.0 | 6.1 | False | False | False |
| 6820 | 121.5 | 77.7 | False | False | False |
| 6915 | 32.1 | 27.2 | False | False | False |
| 6955 | 0.0 | 155.0 | True | False | False |
| 7731 | 7.23 | 7.0 | False | False | False |
| 8162 | 0.0 | 36.75 | True | False | False |
| 9957 | 6.77 | 6.15 | False | False | False |

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

### 2025-05-02 -> 2025-06-02
- period_return: `-0.006916287668378728`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 1569 | 48.7 | 47.95 | False | False | False |
| 2338 | 30.75 | 27.55 | False | False | False |
| 2380 | 3.88 | 3.88 | False | False | False |
| 2718 | 48.35 | 50.5 | False | False | False |
| 3055 | 50.9 | 50.5 | False | False | False |
| 3376 | 193.5 | 173.5 | False | False | False |
| 3426 | 0.0 | 0.0 | True | False | False |
| 4414 | 3.56 | 3.24 | False | False | False |
| 4529 | 4.06 | 3.81 | False | False | False |
| 4927 | 23.55 | 21.4 | False | False | False |
| 5301 | 5.51 | 5.8 | False | False | False |
| 5314 | 64.4 | 54.0 | False | False | False |
| 5701 | 5.38 | 5.17 | False | False | False |
| 6203 | 64.2 | 73.8 | False | False | False |
| 6517 | 66.5 | 80.6 | False | False | False |
| 6668 | 35.3 | 45.0 | False | False | False |
| 8291 | 5.26 | 4.53 | False | False | False |
| 8404 | 21.35 | 21.05 | False | False | False |

### 2025-06-02 -> 2025-07-01
- period_return: `0.11753672108064583`
- calc_status: `ok`

| stock_id | entry_price | exit_price | zero_price | nan_price | extreme_price |
|---|---:|---:|---|---|---|
| 1337 | 5.72 | 5.27 | False | False | False |
| 2314 | 8.02 | 10.35 | False | False | False |
| 2911 | 4.48 | 5.44 | False | False | False |
| 3117 | 3.7 | 0.0 | True | False | False |
| 3178 | 0.0 | 46.6 | True | False | False |
| 4414 | 3.24 | 3.28 | False | False | False |
| 4529 | 3.81 | 4.04 | False | False | False |
| 4584 | 0.0 | 56.5 | True | False | False |
| 5907 | 6.79 | 6.41 | False | False | False |
| 6403 | 5.99 | 6.63 | False | False | False |
| 6417 | 0.0 | 88.9 | True | False | False |
| 6434 | 2.41 | 2.4 | False | False | False |
| 6673 | 6.21 | 7.15 | False | False | False |
| 6730 | 0.0 | 41.35 | True | False | False |
| 6793 | 5.56 | 9.13 | False | False | False |
| 6904 | 0.0 | 131.5 | True | False | False |
| 7732 | 0.0 | 40.85 | True | False | False |
| 7743 | 29.3 | 29.5 | False | False | False |
| 8087 | 0.0 | 41.9 | True | False | False |
| 8291 | 4.53 | 4.9 | False | False | False |

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
