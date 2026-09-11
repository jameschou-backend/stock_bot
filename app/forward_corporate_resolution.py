"""Human-reviewed terms; source-bound exceptions, never accounting overrides."""
from decimal import Decimal as D
import hashlib
from pathlib import Path
import re
from urllib.parse import urlsplit

from app import forward_journal as j, forward_portfolio as p, forward_corporate_audit as a

VERSION = 'corporate-terms-v1'
CASH_CODES = {'policy_revision', 'result_revision', 'amount_conflict', 'payment_unknown',
              'unpaired_dividend', 'missing_entitlement', 'ledger_conflict',
              'unexpected_entitlement', 'unmatched_right', 'payment_due'}


def key(terms):
    return terms['stock_id'], terms['ex_date'], terms['action_type']


def resolutions(rows):
    return {key(r['body']['terms']): r for r in rows if r['kind'] == 'corporate_resolution'}


def fingerprint(terms, evidence_path, clock):
    """Bind all observed rows for this event, not changing retrieval timestamps.

    An unrelated company/event does not invalidate this review. A revised or
    deleted term stays in the evidence set and cannot silently erase a conflict.
    """
    sid, ex, kind = key(terms)
    found, current = {}, {}
    for event in a.source_history(evidence_path):
        b = event['body']
        if b['status'] != 'ok' or j.timestamp(b['retrieved_at']) > clock():
            continue
        ds = b['query'][0]
        relevant = []
        for row in b['rows']:
            if row.get('stock_id') != sid:
                continue
            match = ((ds == a.POLICY and row.get('CashExDividendTradingDate') == ex)
                     or (ds == a.RESULT and row.get('date') == ex)) if kind == 'cash' else (
                         ds == a.SPLIT and row.get('date') == ex)
            if match:
                item = dict(dataset=ds, row=row)
                found[j.digest(item)] = item
                relevant.append(j.digest(item))
        if ds in ((a.POLICY, a.RESULT) if kind == 'cash' else (a.SPLIT,)) and b['query'][1] == sid:
            current[ds] = sorted(set(relevant))
    if not found:
        raise ValueError('找不到這項權益的來源事件；不能建立無依據的豁免')
    return j.digest(dict(observed=sorted(found), current=current)), [found[k] for k in sorted(found)]


def validate_terms(terms):
    c = dict(terms)
    if c.get('action_type') not in ('cash', 'split'):
        raise ValueError('只接受現金股息或可完整核對的分割；減資、配股等仍需另行處理')
    if not isinstance(c.get('stock_id'), str) or not re.fullmatch(r'\d{4}', c['stock_id']):
        raise ValueError('股票代號須四碼')
    ex, payment = a.day(c.get('ex_date')), a.day(c.get('delivery_date'))
    if payment < ex:
        raise ValueError('交付日不可早於除權息日')
    common = {'stock_id', 'ex_date', 'delivery_date', 'action_type'}
    if c['action_type'] == 'cash':
        per = a.number(c.get('cash_per_share'))
        if per <= 0 or set(c) != common | {'cash_per_share'}:
            raise ValueError('現金股息需明確正數每股金額，不接受其他覆寫欄位')
        c['cash_per_share'] = str(per)
    else:
        ratio = a.number(c.get('ratio'))
        halt = a.day(c.get('halt_start'))
        if ratio <= 0 or ratio == 1 or halt >= ex or payment != ex:
            raise ValueError('分割需明確比率、較早的停牌起日，以及恢復交易當日交付；其他時程暫不支援')
        if set(c) != common | {'ratio', 'halt_start'}:
            raise ValueError('分割不接受其他覆寫欄位')
        c['ratio'] = str(ratio)
    return c


def validate_evidence(evidence, clock):
    expected = {'url', 'title', 'published_at', 'text', 'reviewer'}
    if set(evidence) != expected or any(not isinstance(v, str) or not v.strip() for v in evidence.values()):
        raise ValueError('需提供公告連結、標題、含時區發布時間、相關條款及核對人')
    url = urlsplit(evidence['url'])
    if url.scheme != 'https' or not url.hostname or url.username or url.password:
        raise ValueError('請提供不含登入資訊的HTTPS公告連結')
    if j.timestamp(evidence['published_at']) > clock():
        raise ValueError('公告發布時間不能在未來')
    if not 20 <= len(evidence['text']) <= 20000:
        raise ValueError('請保留20至20000字的相關條款供核對')
    return dict(evidence, text_sha256=hashlib.sha256(evidence['text'].encode()).hexdigest(),
                verification='user_reviewed_not_provider_authenticated')


def readiness(terms, rows):
    sid, ex, kind = key(terms)
    s = p.state(rows)
    eligible = a.before_ex(rows, ex)['holdings'].get(sid, {}).get('qty', 0)
    matches = [r for r in s['rights'].values() if r['stock_id'] == sid and r['ex_date'] == ex and r['action_type'] == kind]
    return s, eligible, matches[0] if len(matches) == 1 else None


def apply(report, rows, evidence_path, clock=j.now):
    active = resolutions(rows)
    if not active:
        return report
    result = dict(report, issues=list(report['issues']), events=list(report['events']),
                  resolved_issues=[], resolutions=[], resolution_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    today = str(clock().astimezone(p.TZ).date())

    def issue(sid, ex, code, message, blocking=True):
        result['issues'].append(dict(stock_id=sid, date=ex, code=code, message=message, blocking=blocking))

    for group, event in active.items():
        sid, ex, kind = group
        terms = event['body']['terms']
        if sid not in report['scope']['stock_ids']:
            continue
        try:
            current, _ = fingerprint(terms, evidence_path, clock)
        except ValueError:
            current = None
        if current != event['body']['source_fingerprint']:
            issue(sid, ex, 'resolution_stale', '來源條款有變動，先前核對已失效，需重新預覽與保存')
            result['resolutions'].append(dict(hash=event['hash'], terms=terms, status='來源已變動'))
            continue
        def covered(i):
            return i['stock_id'] == sid and i['date'] == ex and (
                i['code'] in CASH_CODES if kind == 'cash' else (
                    i['code'] == 'structural_action' and i['message'].startswith(a.LABELS[a.SPLIT])))
        result['resolved_issues'] += [dict(i, resolution_hash=event['hash']) for i in result['issues'] if covered(i)]
        result['issues'] = [i for i in result['issues'] if not covered(i)]
        result['resolutions'].append(dict(hash=event['hash'], terms=terms, status='人工條款已保存，入帳仍另行檢查'))
        s, eligible, right = readiness(terms, rows)
        if kind == 'cash':
            result['events'] = [e for e in result['events'] if (e['stock_id'], e['ex_date']) != (sid, ex)]
            result['events'].append(dict(stock_id=sid, ex_date=ex, cash_per_share=terms['cash_per_share'],
                payment_date=terms['delivery_date'], eligible_qty=eligible if ex <= today else None,
                paid=bool(right and right['paid']), terms_source='人工核對公告'))
        else:
            if any(e['stock_id'] == sid and e['ex_date'] == ex for e in report['events']):
                issue(sid, ex, 'mixed_actions', '分割同日另有現金權益，股數基準與計算順序尚不支援')
            for r in rows:
                if r['kind'] == 'fill':
                    order = s['orders'][r['body']['order_id']]
                    executed = str(j.timestamp(r['body']['executed_at']).astimezone(p.TZ).date())
                    if order['stock_id'] == sid and terms['halt_start'] <= executed < ex:
                        issue(sid, ex, 'halt_fill', '分割停牌期間卻有成交紀錄，須另行修復帳本')
            if D(eligible) * D(terms['ratio']) % 1:
                issue(sid, ex, 'fractional_right', '分割後有碎股，不能以整數股數或現金假設放行')
        if ex > today:
            issue(sid, ex, 'reviewed_upcoming', '未到除權息日；條款核對不代表已取得權益', False)
            continue
        if eligible:
            if not right:
                issue(sid, ex, 'reviewed_missing_entitlement', '條款已核對，但權益仍未登錄；不得倒填')
            else:
                same = right['eligible_qty'] == eligible and right['delivery_date'] == terms['delivery_date']
                if kind == 'cash':
                    same &= D(right['cash_per_share']) == D(terms['cash_per_share'])
                    same &= D(right['amount']) == eligible * D(terms['cash_per_share'])
                else:
                    same &= D(right['ratio']) == D(terms['ratio']) and D(right['result_qty']) == eligible * D(terms['ratio'])
                if not same:
                    issue(sid, ex, 'reviewed_ledger_conflict', '既有權益與核對條款不符；保存條款不會修改股數、現金或舊紀錄')
                if not right['paid']:
                    issue(sid, ex, 'reviewed_delivery_pending', '尚缺實際交付紀錄；公告不能證明帳戶已收到股款或新股', kind == 'split')
        elif right:
            issue(sid, ex, 'reviewed_unexpected_right', '除權息前沒有合格股數，卻有權益入帳')
    result['blocked'] = any(i['blocking'] for i in result['issues'])
    return result


def preview(path, terms, evidence, evidence_path=a.PATH, clock=j.now):
    rows = a.account_rows(path, clock)
    terms = validate_terms(terms)
    report = a.inspect(path, evidence_path, clock, rows=rows, resolve=False)
    if terms['stock_id'] not in report['scope']['stock_ids'] or not report['scope']['start'] <= terms['ex_date'] <= report['scope']['end']:
        raise ValueError('核對項目須位於這份帳本的股票及觀察期間')
    if any(i['code'] in {'source_unavailable','invalid_schema','invalid_terms'} for i in report['issues']):
        raise ValueError('請先補齊有效且新鮮的來源，不能用人工核對跳過資料缺漏')
    signature, source_rows = fingerprint(terms, evidence_path, clock)
    previous = resolutions(rows).get(key(terms))
    body = dict(version=VERSION, terms=terms, evidence=validate_evidence(evidence, clock),
                source_fingerprint=signature, source_rows=source_rows,
                account_head=rows[-1]['hash'], supersedes=previous['hash'] if previous else None)
    command = dict(id=j.digest(body)[:32], **body)
    hypothetical = dict(kind='corporate_resolution', body=body, hash='preview')
    return dict(command=command, before=report, after=apply(report, rows + [hypothetical], evidence_path, clock))


def save(path, command, evidence_path=a.PATH, clock=j.now):
    """Optimistic account check and source check in one append transaction."""
    with j.connection(path) as con:
        p.initialize(con, clock)
        rows = j.read_events(con)
        body = {k:v for k,v in command.items() if k != 'id'}
        event_key = 'corporate_resolution:' + command['id']
        existing = next((r for r in rows if r['event_key'] == event_key), None)
        if existing:
            return j.append(con, event_key, 'corporate_resolution', body, clock)
        if body['account_head'] != rows[-1]['hash']:
            raise ValueError('帳本已有新紀錄，請重新預覽核對結果')
        evidence = {k:v for k,v in body['evidence'].items() if k not in {'text_sha256','verification'}}
        # No second connection to this locked account; preview's validation is repeated locally.
        terms = validate_terms(body['terms'])
        checked = a.inspect(path, evidence_path, clock, rows=rows, resolve=False)
        if terms['stock_id'] not in checked['scope']['stock_ids'] or not checked['scope']['start'] <= terms['ex_date'] <= checked['scope']['end']:
            raise ValueError('核對項目已超出帳本觀察範圍')
        if any(i['code'] in {'source_unavailable','invalid_schema','invalid_terms'} for i in checked['issues']):
            raise ValueError('來源無效或已過期，請重新取得並預覽')
        signature, source_rows = fingerprint(terms, evidence_path, clock)
        previous = resolutions(rows).get(key(terms))
        expected = dict(version=VERSION,terms=terms,evidence=validate_evidence(evidence,clock),
            source_fingerprint=signature,source_rows=source_rows,account_head=rows[-1]['hash'],
            supersedes=previous['hash'] if previous else None)
        if body != expected or command['id'] != j.digest(expected)[:32]:
            raise ValueError('條款、來源或預覽已變動，請重新預覽')
        return j.append(con,event_key,'corporate_resolution',body,clock)


def entitlement_draft(path, resolution_hash, evidence_path=a.PATH, clock=j.now):
    """Reviewable JSON only. Frozen engine validates timing, shares and old orders."""
    rows = a.account_rows(path, clock)
    event = next((r for r in resolutions(rows).values() if r['hash'] == resolution_hash), None)
    if not event:
        raise ValueError('請選擇目前版本的核對紀錄')
    c = event['body']['terms']
    signature, _ = fingerprint(c, evidence_path, clock)
    checked = a.inspect(path, evidence_path, clock, rows=rows)
    if signature != event['body']['source_fingerprint'] or any(i['code'] in {
            'source_unavailable', 'invalid_schema', 'invalid_terms', 'mixed_actions', 'halt_fill', 'fractional_right'} for i in checked['issues']):
        raise ValueError('来源或權益仍有未處理問題，不能產生入帳草稿')
    s, eligible, _ = readiness(c, rows)
    command = dict(kind='entitlement', id='reviewed:'+event['hash'][:24], action_id='reviewed:'+event['hash'][:24],
        stock_id=c['stock_id'], action_type=c['action_type'], ex_date=c['ex_date'],
        delivery_date=c['delivery_date'], eligible_qty=eligible, evidence='user_reviewed_terms:'+event['hash'])
    if c['action_type'] == 'cash':
        command.update(cash_per_share=c['cash_per_share'], amount=str(D(c['cash_per_share'])*eligible))
    else:
        n = D(c['ratio']) * eligible
        if n % 1:
            raise ValueError('有碎股，不能產生整數分割草稿')
        command.update(ratio=c['ratio'], result_qty=int(n))
    p._entitlement(command,s,rows,str(clock().astimezone(p.TZ).date()))
    return command
