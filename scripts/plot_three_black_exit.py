#!/usr/bin/env python3
"""Render the published account curves with their verified export hashes."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/three-black-20261001/mpl'))
try:
    import matplotlib
except ImportError as exc:
    raise SystemExit('Install the optional chart dependency: python -m pip install matplotlib') from exc
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates
import pandas as pd
from scripts.research_exit_scenarios import read, sha, write
from skills.three_black_exit import ARMS, NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT)
    if args.output.exists():
        raise ValueError('Preserve prior figure')
    if not args.font.is_file():
        raise ValueError('Provide an installed Chinese font with --font')
    publication = read(args.publication)
    font = FontProperties(fname=str(args.font), size=11)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.1, 1]})
    colors = dict(control='#24384b', three_black='#d57519', benchmark='#8a8f95')
    source_files = [args.publication, Path(__file__)]
    for arm in ARMS:
        p = args.publication.with_suffix('')/(arm+'-daily.csv')
        if sha(p) != publication['exports_sha256'][str(p.resolve().relative_to(ROOT))]:
            raise ValueError('Published daily series changed')
        frame = pd.read_csv(p, parse_dates=['date'])
        source_files.append(p)
        axes[0].plot(frame.date, frame.nav/1e6, label=NAMES[arm], color=colors[arm], lw=1.45)
        axes[1].plot(frame.date, frame.drawdown*100, color=colors[arm], lw=1.25)
    for ax in axes:
        ax.grid(axis='y', alpha=.16)
        ax.axvspan(pd.Timestamp('2023-06-29'), pd.Timestamp('2023-10-31'), color='#bbb', alpha=.18)
    axes[0].set_ylabel('資產（百萬元）', fontproperties=font)
    axes[1].set_ylabel('距歷史高點（%）', fontproperties=font)
    axes[0].legend(prop=font, ncol=3, frameon=False, loc='upper left')
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.suptitle('三黑K出場比較｜100 萬元複利・三檔個股', fontproperties=FontProperties(fname=str(args.font), size=19), x=.075, ha='left')
    fig.text(.075, .922, '2019/01/02–2026/09/09；灰帶為原版最大回撤區間', fontproperties=font, color='#555')
    fig.text(.075, .035, '已扣設定稅費與滑價；日高低價中點為成交代理。已研究歷史，不代表未來或實際成交。', fontproperties=font, color='#555')
    fig.subplots_adjust(left=.08, right=.98, top=.89, bottom=.09, hspace=.13)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    write(args.output.with_suffix('.source.json'), dict(
        source_sha256={str(p.resolve().relative_to(ROOT)):sha(p) for p in source_files},
        font_path=str(args.font), output_sha256=sha(args.output), actual_fill_verified=False))


if __name__ == '__main__':
    main()
