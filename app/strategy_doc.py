"""選股邏輯說明（供 Dashboard 與 API 顯示）"""

from typing import List

from skills.build_features import FEATURE_COLUMNS


def get_selection_logic(config=None) -> str:
    """回傳目前選股邏輯的文字說明。"""
    # 從 config 取得參數（若傳入）
    topn = 20
    min_turnover = 0.5
    market_filter = True
    bear_topn = 10
    stoploss = -0.07
    if config:
        topn = getattr(config, "topn", 20)
        min_turnover = getattr(config, "min_avg_turnover", 0.5)
        market_filter = getattr(config, "market_filter_enabled", True)
        bear_topn = getattr(config, "market_filter_bear_topn", 10)
        stoploss = getattr(config, "stoploss_pct", -0.07)

    features_desc = _format_features(FEATURE_COLUMNS)
    return f"""## 選股邏輯概要

### 1. 資料與模型
- **特徵來源**：價格、三大法人、融資融券（raw_prices, raw_institutional, raw_margin_short）
- **模型**：LightGBM / sklearn GBR，以未來報酬率（label_horizon_days 日）為排序目標
- **訓練**：每季（週一）重新訓練，Spearman 相關性作為評估指標

### 2. 過濾條件
- **股票 Universe**：`security_type=stock`、`is_listed=True`（排除 ETF、權證）
- **流動性**：20 日平均成交值 ≥ {min_turnover} 億元（過濾冷門股）
- **大盤過濾**：{'啟用' if market_filter else '關閉'} — 空頭市場（60 日均線下）時減碼至 {bear_topn} 檔

### 3. 選股流程
1. 從 features 表取最近 {(config.fallback_days if config else 10)} 個交易日作為候選
2. 依流動性、Universe 過濾
3. 模型對特徵預測得分（score）
4. 依 score 由高到低取 Top-{topn} 檔（空頭時取 Top-{bear_topn}）
5. 寫入 picks 表，供回測/實盤使用

### 4. 特徵列表（{len(FEATURE_COLUMNS)} 維）
{features_desc}

### 5. 回測設定
- **持有週期**：每月初再平衡，持有一個月或觸發停損
- **停損**：單檔跌破 {stoploss*100:.0f}% 出場
- **交易成本**：來回約 0.585%（手續費 + 證交稅）
"""


def _format_features(cols: List[str]) -> str:
    """將特徵列表格式化成 Markdown 清單"""
    groups = {
        "動能": ["ret_5", "ret_10", "ret_20", "ret_60"],
        "均線": ["ma_5", "ma_20", "ma_60"],
        "技術": ["bias_20", "vol_20", "vol_ratio_20", "rsi_14", "macd_hist", "kd_k", "kd_d"],
        "法人": ["foreign_net_5", "foreign_net_20", "trust_net_5", "trust_net_20", "dealer_net_5", "dealer_net_20"],
        "籌碼": ["margin_balance_chg_5", "margin_balance_chg_20", "short_balance_chg_5", "short_balance_chg_20", "margin_short_ratio"],
        "大盤": ["market_rel_ret_20"],
    }
    lines = []
    for group_name, group_cols in groups.items():
        found = [c for c in group_cols if c in cols]
        if found:
            lines.append(f"- **{group_name}**：{', '.join(found)}")
    remaining = [c for c in cols if not any(c in g for g in groups.values())]
    if remaining:
        lines.append(f"- **其他**：{', '.join(remaining)}")
    return "\n".join(lines) if lines else ", ".join(cols)
