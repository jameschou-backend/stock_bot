# 高報酬個股版本重新回測（2026-09-28）

已完成：三個不同的純個股版本，正常 / 壓力各一組；另外重跑五檔原對照與 0050，共 10 個完整帳戶。兩次獨立離線重播逐欄一致，原有五檔完整帳戶也完全重現。

正常情境以三檔成交金額排序最高：**+320.43%，100 萬變 420.43 萬**，高於同成本基準 0050 的 +238.96%。但是同策略在滑價加倍、可成交量減半後只剩 **+62.30%**，低於壓力基準 +232.69%。這輪支持「正常假設曾勝出」，不支持「穩健勝大盤或已可實戰」。

## 同期完整帳戶排行

期間 2022-01-03～2026-09-09，1136 市場日；本金 100 萬複利，均為扣成本累積報酬、非年化。期末包括未賣持股及應收權息，未假設末日全數變現。

| 版本 | 正常淨報酬 | 正常期末資產 | 正常最大回撤 | 壓力淨報酬 | 壓力最大回撤 |
|---|---:|---:|---:|---:|---:|
| 三檔：成交金額排序 | 320.43% | 4,204,303 元 | -31.60% | 62.30% | -24.73% |
| 三檔：成交金額 / 波動排序 | 260.55% | 3,605,455 元 | -30.44% | 94.19% | -24.94% |
| 0050 持有基準 | 238.96% | 3,389,588 元 | -33.92% | 232.69% | -34.03% |
| 五檔：成交金額排序（原對照） | 153.30% | 2,532,993 元 | -29.27% | 153.58% | -27.51% |
| 三檔：原始優先順序 | 135.74% | 2,357,421 元 | -34.53% | 70.41% | -34.86% |

## 舊高報酬映射

| 舊封存報酬 | 原設計 | 本輪重測 |
|---|---|---|
| 1193.66% | 成交金額排序、僅鎖未用預算 | 統一 C/S/U 後與下一列合併：320.43% |
| 1111.00% | 成交金額排序、舊資源重用 | 320.43% |
| 1142.93% | 成交金額 / 波動排序 | 260.55% |
| 541.90% | 原始順序、閒錢現金 | 135.74% |
| 710.81% / 705.92% | 閒錢買 0050 / 趨勢 ON 買 0050 | 依純個股要求未重新啟用，不列成本輪績效 |

舊研究有 458 候選，本輪是修正後固定 454 候選；還原、停市、歷史身分與成交口徑亦已修正。新舊差額混合多項修正，不能把全部下降歸因於零股、延一天或單一因素。

## 最高版本逐年（正常）

| 年度 | 三檔成交金額 | 同期 0050 |
|---|---:|---:|
| 2022 | -14.29% | -22.98% |
| 2023 | 47.61% | 27.42% |
| 2024 | 2.44% | 48.34% |
| 2025 | 48.75% | 36.77% |
| 2026（至 9/9） | 118.09% | 70.22% |

252 市場日滾動勝出比例 54.41%，共 884 個重疊窗口，並非獨立或未見測試。正常、壓力之間有 14 個只在正常買入的事件、11 個只在壓力買入的事件、42 個共同事件買入數量不同。因此壓力報酬下降同時包括容量、部位、退出與後續複利改變，不能視為同一帳本單純多扣費用。

## 成本與交易數

| 版本 | 正常總成本 | 正常買 / 賣成交筆數 | 壓力總成本 | 壓力買 / 賣成交筆數 |
|---|---:|---:|---:|---:|
| 三檔：成交金額排序 | 337,842 | 97 / 138 | 272,062 | 85 / 136 |
| 三檔：成交金額 / 波動排序 | 314,007 | 95 / 135 | 274,029 | 83 / 134 |
| 三檔：原始優先順序 | 280,051 | 98 / 157 | 253,898 | 81 / 148 |
| 五檔：成交金額排序（原對照） | 313,040 | 149 / 192 | 412,898 | 142 / 241 |
| 0050 持有基準 | 7,333 | 41 / 0 | 12,542 | 44 / 0 |

## 執行及證據限制

- T 收盤產生訊號，T+1 委託；沒有額外再延一天。正常每邊滑價 45bps、容量 1%；壓力每邊 90bps、容量 0.5%。佣金 0.1425%、每通道最低 20 元；個股賣出稅 0.3%，ETF 0.1%。
- 普通盤買單使用開盤成交批次推估；零股使用獨立零股全日價量、高買低賣估計與嚴格價格穿越。零股不是逐筆撮合證明，也未驗證撤單延遲。
- 開盤前鎖定現金、名額及未用預算。不以當日賣出所得或稍後拒單資金回頭買開盤。已出場但尚未清完的零股資產仍保留，受 5% 殘餘曝險上限控制。
- 同歷史反覆研究，非未見樣本；歷史名冊重建仍不等同完整歷史全市場。live_qualified=false、unseen_validation=false。未恢復排程、未下單、未更換正式策略。

## 重播與保存

本輪資料準備新增 27 次 FinMind、2 次官方零股 HTTP；FinMind 共用準備計數 186→213 / 300，官方累計 6→8 / 119，沿用共享小時限額。兩次正式回播各 0 網路請求，耗時 134.845 秒及 130.479 秒（各 10 帳戶）。

- 驗收：make test 4032 passed、36 warnings；make pipeline exit 0。make api 因現有服務占用 8000 退出，保留原服務，health / picks / models / jobs 四端點均 HTTP 200。
- 實驗登記已補記本輪 40 次執行（包含預檢、準備及兩次離線重播）；這些重播不是 40 個獨立策略或未見樣本。
- 新增測試包含 3 檔資金核帳、5 檔完整帳戶中性對照、未來價格 / 成交量不能改變先前排序及預算，以及被竄改預算必須被核帳拒絕。
- [正式摘要與來源雜湊](/Users/james.chou/JamesProject/stock_bot/artifacts/forward_simulation/high_return_revalidation_20260928.json)
- [逐年、個股損益及正常 / 壓力路徑比較](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/analysis.json)

每個帳戶均包含完整買賣、拒單、預先委託、每日淨值、現金流水、持股與公司行動：

- [benchmark_normal](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/benchmark_normal.json)
- [benchmark_stress](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/benchmark_stress.json)
- [capacity_normal](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/capacity_normal.json)
- [capacity_stress](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/capacity_stress.json)
- [capacity_vol_normal](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/capacity_vol_normal.json)
- [capacity_vol_stress](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/capacity_vol_stress.json)
- [control5_normal](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/control5_normal.json)
- [control5_stress](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/control5_stress.json)
- [original_normal](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/original_normal.json)
- [original_stress](/Users/james.chou/JamesProject/stock_bot/.cache/high-return-20260928/final-a/cases/original_stress.json)

本機 .cache 的完整帳戶不是異地備份；Git 保存摘要、雜湊、方法與程式。

```sh
python scripts/research_high_return.py --output .cache/high-return-20260928/new-offline-run
python scripts/research_high_return.py --compare .cache/high-return-20260928/final-a .cache/high-return-20260928/final-b --output /tmp/high-return-verified.json
```
