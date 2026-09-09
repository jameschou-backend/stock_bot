"""Explainable headline triage; no return data, LLM, orders, or profitability score."""
from __future__ import annotations

from collections import Counter
import hashlib
import re
import unicodedata
from urllib.parse import urlsplit

VERSION = 'headline-rules-1'
THEMES = {
    'memory': ('記憶體', r'記憶體|DRAM|NAND|HBM|DDR[3456]'),
    'passive': ('被動元件', r'被動元件|MLCC|電容|電阻'),
    'leo': ('低軌／衛星通訊', r'低軌|衛星|Starlink|SpaceX'),
    'cooling': ('散熱／液冷', r'散熱|液冷|水冷'),
    'power': ('電力設備', r'重電|變壓器|電網|HVDC|高壓直流'),
    'packaging': ('先進封裝', r'先進封裝|CoWoS|玻璃基板|FOPLP'),
    'ai_server': ('AI 伺服器', r'AI\s*伺服器|GB[23]00|Blackwell|Rubin'),
    'photonics': ('光通訊／矽光子', r'矽光子|光通訊|\bCPO\b'),
    'robotics': ('機器人', r'機器人|機械手臂'),
    'drones': ('無人機', r'無人機'),
    'storage': ('儲能／備援電力', r'儲能|\bBBU\b|備援電力'),
}
EVENTS = {
    'supply': ('供給／庫存', r'缺貨|供不應求|減產|停產|供給吃緊|庫存去化|去庫存'),
    'pricing': ('報價／漲價', r'漲價|調漲|(?:報價|售價).{0,8}(?:上揚|上漲|攀升|走高)'),
    'orders': ('接單／客戶', r'接單|訂單|獲單|大單|客戶認證'),
    'production': ('量產／出貨', r'量產|出貨|投產|送樣|拉貨'),
    'earnings': ('營收／獲利', r'轉盈|轉虧為盈|(?:營收|毛利|獲利).{0,12}(?:增|新高|成長|改善)'),
}
NEGATIVE = re.compile(r'衰退|下滑|虧損|砍單|訂單.{0,6}(?:砍|減)|遭砍|取消|延後|延期|降價|下修|不如預期|否認|澄清|假消息|尚未|未量產|未出貨|未獲單|並未|不確定|疑慮|恐')
EXPECTATION = re.compile(r'展望|預期|有望|可望|將|傳出|傳言|傳(?=停產|減產|漲價)|看好|可能|預估|目標價|估計|概念|題材')
PRICE = re.compile(r'大漲|漲停|飆漲|亮燈|(?:股價|盤中).{0,8}(?:創高|新高|噴|揚)|強漲|下殺|跌停|爆量|目標價')
STATUS = {'operating_clue': '營運線索待核對', 'expectation': '預期／概念敘事',
          'negative_or_mixed': '負面或正反並存', 'price_commentary': '股價／追漲報導',
          'topic_only': '僅題材關聯'}
PATTERNS = {key: re.compile(pattern, re.I) for key, (_, pattern) in THEMES.items()}
EVENT_PATTERNS = {key: re.compile(pattern, re.I) for key, (_, pattern) in EVENTS.items()}


def safe_link(value):
    value = str(value or '').strip()
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme in ('http', 'https') and parsed.hostname and not parsed.username else ''
    except ValueError:
        return ''


def clean_title(value, source=''):
    title = unicodedata.normalize('NFKC', str(value or '')).strip()
    # Syndicated headlines routinely differ only in publisher suffix or punctuation.
    if ' - ' in title:
        body, tail = title.rsplit(' - ', 1)
        if tail.strip() == unicodedata.normalize('NFKC', str(source)).strip():
            title = body
    return title


def title_key(title):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', title).casefold()


def classify(title):
    themes = [key for key, pattern in PATTERNS.items() if pattern.search(title)]
    events = [key for key, pattern in EVENT_PATTERNS.items() if pattern.search(title)]
    negative = bool(NEGATIVE.search(title))
    expectation = bool(EXPECTATION.search(title))
    chase = bool(PRICE.search(title))
    templated = bool(re.search(r'【.*(?:即時新聞|投資快訊).*】', title))
    status = ('negative_or_mixed' if negative else 'price_commentary' if chase and (not events or templated)
              else 'expectation' if expectation and events else 'operating_clue' if events else
              'price_commentary' if chase else 'topic_only')
    return {'themes': themes, 'events': events, 'negative': negative,
            'expectation': expectation, 'price_commentary': chase, 'status': status,
            'matched_terms': sorted({m.group() for p in [*PATTERNS.values(), *EVENT_PATTERNS.values(),
                                                         NEGATIVE, EXPECTATION, PRICE] for m in p.finditer(title)})}


def analyze(rows, names):
    """Deduplicate titles across related stocks and publishers, then classify once.

    A FinMind stock tag is a research association, never proof of a beneficiary.
    Distinct sources are publishers, NOT independent confirmations of a story.
    """
    stories = {}
    invalid = 0
    for row in rows:
        sid = str(row.get('stock_id', ''))
        title = clean_title(row.get('title'), row.get('source'))
        if not re.fullmatch(r'[0-9]{4}', sid) or sid not in names or not title:
            invalid += 1
            continue
        key = title_key(title)
        if not key:
            invalid += 1
            continue
        stamp = str(row['provider_datetime'])
        story = stories.setdefault(key, {'id': hashlib.sha256(key.encode()).hexdigest()[:20],
            'title': title, 'provider_datetime': stamp, 'source_date': stamp[:10],
            'first_recorded_at': row['first_recorded_at'], 'stock_ids': set(), 'sources': set(),
            'links': set(), 'raw_rows': 0})
        story['stock_ids'].add(sid)
        if row.get('source'): story['sources'].add(str(row['source']))
        link = safe_link(row.get('link'))
        if link: story['links'].add(link)
        story['provider_datetime'] = min(story['provider_datetime'], stamp)
        story['source_date'] = story['provider_datetime'][:10]
        story['first_recorded_at'] = min(story['first_recorded_at'], row['first_recorded_at'])
        story['raw_rows'] += 1
    classified = []
    unknown = Counter()
    for story in stories.values():
        for key in ('stock_ids', 'sources', 'links'): story[key] = sorted(story[key])
        story.update(classify(story['title']))
        explicit = []
        for sid in story['stock_ids']:
            name = re.sub(r'\*|-KY$', '', names[sid])
            if (len(name) >= 2 and name in story['title']) or re.search(rf'(?<!\d){sid}(?!\d)', story['title']):
                explicit.append(sid)
        story['headline_named_ids'] = explicit
        story['evidence_level'] = 'headline_only_unverified'
        if not story['themes']:
            for phrase in re.findall(r'([\u4e00-\u9fffA-Za-z0-9]{2,8})題材', story['title']):
                unknown[phrase] += 1
        classified.append(story)
    classified.sort(key=lambda s: (s['provider_datetime'], s['id']))
    themes = []
    for tid, (name, _) in THEMES.items():
        group = [s for s in classified if tid in s['themes']]
        if not group: continue
        members = sorted({sid for s in group for sid in s['stock_ids']})
        counts = Counter(s['status'] for s in group)
        themes.append({'id': tid, 'name': name, 'articles': len(group),
                       'operating_clues': counts['operating_clue'], 'expectations': counts['expectation'],
                       'negative_or_mixed': counts['negative_or_mixed'], 'price_commentary': counts['price_commentary'],
                       'publisher_count': len({p for s in group for p in s['sources']}),
                       'stock_ids': members, 'first_in_window': group[0]['source_date'],
                       'last_in_window': group[-1]['source_date']})
    themes.sort(key=lambda t: (-t['operating_clues'], -t['articles'], t['id']))
    return {'rules_version': VERSION, 'input_rows': len(rows), 'invalid_or_non_stock_rows': invalid,
            'unique_articles': len(classified), 'duplicates_collapsed': len(rows)-invalid-len(classified),
            'themes': themes, 'stories': classified,
            'unclassified_topic_phrases': [{'phrase': p, 'articles': n} for p, n in unknown.most_common(15)]}
