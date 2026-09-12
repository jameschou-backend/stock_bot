# 櫃買中心 TLS 憑證鏈診斷

2026/9/13，Python requests與curl均無法驗證www.tpex.org.tw憑證。openssl檢查顯示伺服器只提供站台憑證，缺TWCA SSL Certification Authority中繼鏈。

新增 `scripts/prepare_tpex_tls_chain.py`：從站台憑證AIA指向的TWCA簽發者網址，以HTTPS取得中繼憑證；使用既有certifi根憑證、openssl `-untrusted` 中繼鏈及 `-verify_hostname www.tpex.org.tw` 驗證。成功才建立本地bundle，不新增未知根憑證、不改系統信任、不修改.env，不使用verify=False。

```sh
python scripts/prepare_tpex_tls_chain.py --output .cache/tpex-tls-check
REQUESTS_CA_BUNDLE="$PWD/.cache/tpex-tls-check/bundle.pem" python scripts/research_cash_risk.py --prepare
```

此設定僅作用於該次程序。腳本需Python requests、certifi、cryptography與openssl；cryptography缺少時會提示安裝。驗證或憑證更新失敗應停止，不沿用未驗證的新鏈。

實際驗證通過後，櫃買歷史零股端點可回HTTP200及日期相符JSON。本輪後續仍出現502／520或傳輸截斷，因此TLS鏈補齊不等於官方服務可靠性已完全恢復。官方來源pipeline在有效鏈下仍曾因Response ended prematurely失敗；明確指定FinMind來源的pipeline驗收通過。

兩項測試涵蓋成功鏈與驗證失敗不能建立bundle。公開憑證hash、簽發者與驗證結果保存在 `artifacts/forward_simulation/tpex_tls_chain_20260913.json`。
