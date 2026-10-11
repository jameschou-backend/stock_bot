"""8927 settlement reconstruction with explicitly secondary delivery evidence."""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import normalized

KEY = '8927-2024-09-06'
RATES = (Decimal('0.02996241524'), Decimal('0.09987471341'))
NOTE = '8927／2024 配股：議事錄及年度財報原檔已核對；2024/10/31 普通股發放與掛牌日依公司及櫃買公告的媒體轉載交叉核對，尚缺公告原站回應。當日休市不成交；分項配股與合併記帳須核對股數及畸零毛額相同。畸零款淨額與支付日未證實，不列可用現金。'
URLS = {
    'minutes': 'https://www.nspco.com.tw/wp-content/uploads/2020/05/1130621%E5%8C%97%E5%9F%BA%E5%9C%8B%E9%9A%9B%E8%AD%B0%E4%BA%8B%E9%8C%84%E4%B8%8A%E5%82%B3.pdf',
    'financial': 'https://www.nspco.com.tw/wp-content/uploads/2020/05/%E5%8C%97%E5%9F%BA113Q4-%E5%90%88%E4%BD%B5%E8%B2%A1%E5%A0%B1%E4%B8%8A%E5%82%B3-1.pdf',
    'listing': 'https://m.cnyes.com/news/print/5749526',
    'delivery': 'https://www.moneydj.com/kmdj/news/newsviewer.aspx?a=696dc924-e00e-4542-b85a-1935fbd8c9f6',
}
FACTS = {
    'minutes': ['轉增資發行普通股9,793,258股', '普通股32,644,192股,每股面額10元',
        '按面額折發現金,計算至元止(元以下捨去)', '前項畸零股款將充抵股東劃撥集保之費用', '本案照原案表決通過'],
    'financial': ['盈餘轉增資9,793', '資本公積轉增資32,644', '實際每千股無償配發99.87股,法定登記程序均已辦理完竣'],
    'listing': ['上櫃股票代號:8927', '普通股42,437,450股', '含員工酬勞轉增資0股',
        '上櫃股票開始買賣日期:民國113年10月31日', '上櫃股票權利義務:與原已發行之股份相同'],
    'delivery': ['普通股42,437,450股', '於113年10月31日起發放並上櫃買賣',
        '該增資新股於發放日即撥入貴股東於證券商開設之集保帳戶'],
}
EXPECTED = dict(shares_per_share=.12983712865, pay_date='2024-10-31', ordinary_share_available_date='2024-10-31',
    fractional_cash_per_share=10, fractional_cash_rounding='floor_ntd', fractional_cash_pay_date=None,
    fractional_cash_payment_date_verified=False, fractional_cash_net_amount_verified=False,
    fractional_cash_available_for_trading=False,
    fractional_cash_treatment='gross_undated_receivable_upper_bound_net_unknown',
    use_scope='account_settlement_only_not_selection')


def load_completion(root, document):
    root, document = Path(root).resolve(), Path(document).resolve(); sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != 'all_entry_northstar_completion_v1' or doc.get('key') != KEY
            or doc.get('strategy_parameters_changed') is not False or doc.get('live_qualified') is not False
            or doc.get('evidence_note') != NOTE or set(doc.get('evidence', {})) != set(URLS)):
        raise ValueError('Northstar evidence scope differs')
    row, refs = doc['override'], doc['source_sha256']
    if (any(row.get(k) != v for k, v in EXPECTED.items()) or 'certificate_delivery_date' in row
            or set(row['evidence_files']) != set(refs)):
        raise ValueError('Northstar ordinary entitlement or cash differs')
    for name, sha in refs.items(): sources.bind(name, sha)
    for kind, item in doc['evidence'].items():
        if any(item[k] not in refs for k in ('path', 'receipt_path')):
            raise ValueError('Northstar source escapes inventory')
        path = root / item['path']; meta = json.loads((root / item['receipt_path']).read_text())
        if (meta.get('url') != URLS[kind] or meta.get('final_url') != meta['url']
                or meta.get('http_status') != 200 or meta.get('automatic_redirects_disabled') is not True
                or meta.get('sha256') != refs[item['path']]):
            raise ValueError('Northstar source response differs')
        if kind in ('minutes', 'financial'):
            if (not path.read_bytes().startswith(b'%PDF') or item.get('text_path') not in refs
                    or item.get('reviewed_pdf_pages') != ([15,16] if kind == 'minutes' else [56])
                    or item.get('source_role') != 'primary_issuer_pdf'):
                raise ValueError('Northstar issuer original is not reviewed')
            text = (root / item['text_path']).read_text()
        else:
            if item.get('source_role') != 'secondary_republication_not_primary_response':
                raise ValueError('Northstar republication cannot be promoted to primary evidence')
            text = BeautifulSoup(path.read_bytes(), 'html.parser').get_text()
        if any(normalized(fact) not in normalized(text) for fact in FACTS[kind]):
            raise ValueError('Northstar source facts differ: ' + kind)
    feed = doc['existing_dividend_feed']
    if feed not in refs: raise ValueError('Northstar feed is unbound')
    df = pd.read_parquet(root / feed)
    selected = df.loc[df.stock_id.eq('8927') & df.StockExDividendTradingDate.eq('2024-09-06')]
    if len(selected) != 1: raise ValueError('Northstar final ex-date differs')
    values = selected.iloc[0]
    rates = tuple(Decimal(str(values[k])) / 10 for k in ('StockEarningsDistribution', 'StockStatutorySurplus'))
    if (rates != RATES or int(values.ParticipateDistributionOfTotalShares) != 326851421
            or any(abs(rate*326851421-quantity) >= 1 for rate,quantity in zip(rates,(9793258,32644192)))):
        raise ValueError('Northstar final ratio or issue quantity differs')
    return deepcopy(row), dict(sources.used)


def guard_component_rounding(case):
    for event in case.get('account', {}).get('corporate_actions', []):
        if (event.get('stock_id'), event.get('date'), event.get('kind')) != ('8927','2024-09-06','stock_dividend'):
            continue
        quantity = Decimal(event['entitled_qty']); parts = [quantity * rate for rate in RATES]
        whole = lambda value: value.to_integral_value(rounding=ROUND_FLOOR)
        combined = sum(parts)
        if (sum(whole(p) for p in parts) != whole(combined)
                or sum(whole((p-whole(p))*10) for p in parts) != whole((combined-whole(combined))*10)):
            raise ValueError('Northstar component rounding differs; separate settlement required')
        if event['whole_new_shares'] != int(whole(combined)):
            raise ValueError('Northstar credited share quantity differs')


def install_completion(runner, provider, document, root):
    row, refs = load_completion(root, document)
    if KEY in provider.corporate_overrides and provider.corporate_overrides[KEY] != row:
        raise ValueError('Northstar completion would rewrite existing settlement')
    provider.corporate_overrides = dict(provider.corporate_overrides, **{KEY: row})
    writer = runner.write
    def write(path, value):
        if isinstance(value, dict) and value.get('completed') is True and 'account' in value:
            guard_component_rounding(value)
        return writer(path, value)
    runner.write = write
    return refs
