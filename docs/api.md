# API Endpoints

Base URL: `http://localhost:8000`

## GET /health
- 回傳服務狀態。

## GET /picks?date=YYYY-MM-DD
- 取得指定日期（預設最新）TopN 選股結果。

## GET /stock/{stock_id}?date=YYYY-MM-DD
- 取得指定股票在指定日期（預設最新）資料：價格、法人、特徵、當日 pick。

## GET /models
- 取得最新 20 筆模型版本。

## GET /jobs?limit=50
- 取得最新 job logs。
