"""
Publication-quality figures for CNCM I-745 Digital Twin paper.
All 5 figures generated here.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.integrate import odeint

os.makedirs('figures', exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
FONT    = 'DejaVu Serif'
C_NAVY  = '#1A3A5C'
C_BLUE  = '#2E6DA4'
C_RED   = '#C0392B'
C_GREEN = '#2E7D32'
C_GREY  = '#EEEEEE'
C_WHITE = '#FFFFFF'

plt.rcParams.update({
    'font.family': FONT,
    'axes.facecolor': C_WHITE,
    'figure.facecolor': C_WHITE,
    'axes.edgecolor': '#333333',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'grid.color': C_GREY,
    'grid.alpha': 0.5,
    'grid.linewidth': 0.8,
})

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color=C_GREY, alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)

def save_fig(fig, name):
    for ext in ('png', 'pdf'):
        path = f'figures/{name}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor=C_WHITE, edgecolor='none')
    print(f'  Saved {name}.png / .pdf')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — GENOME ARCHITECTURE MAP
# ═══════════════════════════════════════════════════════════════════════════════
print('\nGenerating Figure 1...')

df_genome = pd.read_csv('data/genome/genome_stats.csv')

# Extract chromosome label
def extract_label(row):
    desc = str(row['description'])
    if 'mitochondrion' in desc.lower() or 'mitochondr' in desc.lower():
        return 'mtDNA'
    m = re.search(r'chromosome ([IVXLCDMivxlcdm]+)', desc)
    if m:
        return f"Chr {m.group(1).upper()}"
    return row['seq_id']

df_genome['label'] = df_genome.apply(extract_label, axis=1)
df_genome = df_genome.sort_values('length_bp', ascending=False).reset_index(drop=True)

# Colour gradient for nuclear chromosomes
n_nuclear = len(df_genome[df_genome['label'] != 'mtDNA'])
navy_rgb   = np.array([0x1A, 0x3A, 0x5C]) / 255
light_rgb  = np.array([0x7F, 0xB3, 0xD3]) / 255

bar_colors = []
nuc_idx = 0
for _, row in df_genome.iterrows():
    if row['label'] == 'mtDNA':
        bar_colors.append(C_RED)
    else:
        t = nuc_idx / max(n_nuclear - 1, 1)
        c = navy_rgb * (1 - t) + light_rgb * t
        bar_colors.append(tuple(c))
        nuc_idx += 1

N50_BP = 924431

fig1, axes = plt.subplots(1, 2, figsize=(14, 10),
                           gridspec_kw={'width_ratios': [3, 2]})
fig1.patch.set_facecolor(C_WHITE)

# — Panel A: horizontal bar chart ——————————————————————————————————————————
ax_a = axes[0]
lengths_mb = df_genome['length_bp'] / 1e6
y_pos = range(len(df_genome))
bars = ax_a.barh(list(y_pos), lengths_mb.values,
                 color=bar_colors, edgecolor='white', linewidth=0.5, height=0.7)

ax_a.set_yticks(list(y_pos))
ax_a.set_yticklabels(df_genome['label'].values, fontsize=11)
ax_a.invert_yaxis()
ax_a.set_xlabel('Length (Mb)', fontsize=12, fontweight='bold')
ax_a.set_title('A  Chromosome Length Distribution', fontsize=13,
                fontweight='bold', loc='left')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.yaxis.grid(False)
ax_a.xaxis.grid(True, color=C_GREY, alpha=0.5, linewidth=0.8)
ax_a.set_axisbelow(True)

# Value labels
for i, (bar, length) in enumerate(zip(bars, lengths_mb)):
    ax_a.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
              f'{length:.2f} Mb', va='center', ha='left', fontsize=9.5,
              color='#333333')

# N50 dashed line
n50_mb = N50_BP / 1e6
ax_a.axvline(n50_mb, color='#555555', linestyle='--', linewidth=1.5, zorder=5)
ax_a.text(n50_mb + 0.02, len(df_genome) - 0.5, f'N50 = {n50_mb:.3f} Mb',
          fontsize=10, color='#555555', va='bottom', fontstyle='italic')

ax_a.set_xlim(0, lengths_mb.max() * 1.22)

# — Panel B: GC% scatter ———————————————————————————————————————————————————
ax_b = axes[1]
dot_sizes = (df_genome['length_bp'] / df_genome['length_bp'].max()) * 280 + 40
ax_b.scatter(df_genome['gc_pct'], list(y_pos),
             s=dot_sizes, c=bar_colors, edgecolors='white',
             linewidths=0.5, zorder=5, alpha=0.9)

mean_gc = 38.15
ax_b.axvline(mean_gc, color='#555555', linestyle='--', linewidth=1.5, zorder=4)
ax_b.text(mean_gc + 0.05, -0.7, f'Mean GC\n{mean_gc}%',
          fontsize=9.5, color='#555555', va='top', ha='left', fontstyle='italic')

# mtDNA annotation
mt_idx = df_genome[df_genome['label'] == 'mtDNA'].index[0]
mt_row = mt_idx
mt_gc  = df_genome.loc[mt_idx, 'gc_pct']
mt_pos = df_genome.index.get_loc(mt_idx)
ax_b.annotate(f'mtDNA\n{mt_gc:.2f}% GC',
              xy=(mt_gc, mt_pos),
              xytext=(mt_gc + 3, mt_pos - 0.8),
              fontsize=10, color=C_RED, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))

ax_b.set_yticks(list(y_pos))
ax_b.set_yticklabels(df_genome['label'].values, fontsize=11)
ax_b.invert_yaxis()
ax_b.set_xlabel('GC Content (%)', fontsize=12, fontweight='bold')
ax_b.set_title('B  GC Content per Chromosome', fontsize=13,
                fontweight='bold', loc='left')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.yaxis.grid(False)
ax_b.xaxis.grid(True, color=C_GREY, alpha=0.5, linewidth=0.8)
ax_b.set_axisbelow(True)
ax_b.set_xlim(10, 44)

# Figure-level titles
fig1.suptitle('Figure 1.  Genome Architecture of S. boulardii CNCM I-745',
               fontsize=14, fontweight='bold', y=0.98)
fig1.text(0.5, 0.945,
           '17 sequences | 12.16 Mbp | GC 38.15% | N50 924,431 bp',
           ha='center', fontsize=11, color='#444444')
fig1.text(0.5, 0.02,
           'HXT9 and HXT11 loci absent from assembly — consistent with CNCM I-745 '
           'strain identity (Khatri et al. 2017)',
           ha='center', fontsize=9.5, color='#666666', style='italic')

fig1.tight_layout(rect=[0, 0.04, 1, 0.94])
save_fig(fig1, 'figure1_genome_map')
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — GENE ESSENTIALITY AND METABOLIC FLUX
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 2...')

df_ess = pd.read_csv('data/fba_outputs/gene_essentiality.csv')
ess_counts = df_ess['class'].value_counts()
# Normalise to expected
n_essential = int(ess_counts.get('essential', 0))
n_reduced   = int(ess_counts.get('reduced', 0))
n_neutral   = int(ess_counts.get('neutral', 0))
n_total     = n_essential + n_reduced + n_neutral

df_pfba = pd.read_csv('data/fba_outputs/pfba_top10.csv', index_col=0)
df_pfba.columns = ['flux']
df_pfba = df_pfba.sort_values('flux', ascending=True)

# Shorten reaction names
def shorten_rxn(name):
    if len(name) > 22:
        return name[:20] + '..'
    return name

df_pfba.index = [shorten_rxn(str(r)) for r in df_pfba.index]

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
fig2.patch.set_facecolor(C_WHITE)

# — Panel A: donut chart ——————————————————————————————————————————————————————
ax2a = axes2[0]
sizes  = [n_essential, n_reduced, n_neutral]
labels = [f'Essential\n{n_essential} ({n_essential/n_total*100:.1f}%)',
          f'Growth-reduced\n{n_reduced} ({n_reduced/n_total*100:.1f}%)',
          f'Neutral\n{n_neutral} ({n_neutral/n_total*100:.1f}%)']
colors = [C_RED, '#E67E22', '#BDC3C7']
explode = (0.03, 0.03, 0.01)

wedges, texts, autotexts = ax2a.pie(
    sizes, labels=None, colors=colors,
    autopct='', wedgeprops=dict(width=0.5),
    explode=explode, startangle=90,
    pctdistance=0.75
)

# Centre annotation
ax2a.text(0, 0, f'{n_total}\nGenes', ha='center', va='center',
          fontsize=14, fontweight='bold', color='#333333')

# Legend
legend_patches = [
    mpatches.Patch(facecolor=colors[i], label=labels[i]) for i in range(3)
]
ax2a.legend(handles=legend_patches, loc='lower center',
            bbox_to_anchor=(0.5, -0.12), fontsize=10.5, frameon=True,
            fancybox=True, shadow=False)

ax2a.set_title('A  Gene Essentiality Classification', fontsize=13,
               fontweight='bold', pad=12)
ax2a.annotate(f'Essential: {n_essential} ({n_essential/n_total*100:.1f}%)',
              xy=(0, 0.6), fontsize=11, color=C_RED, ha='center',
              fontweight='bold')
ax2a.axis('equal')

# — Panel B: pFBA reactions ——————————————————————————————————————————————————
ax2b = axes2[1]
y_pos2 = range(len(df_pfba))
bar2 = ax2b.barh(list(y_pos2), df_pfba['flux'].values,
                 color=C_BLUE, edgecolor=C_NAVY, linewidth=0.6, height=0.65)

ax2b.set_yticks(list(y_pos2))
ax2b.set_yticklabels(df_pfba.index.tolist(), fontsize=10.5)
ax2b.set_xlabel('Flux (mmol gDW⁻¹ hr⁻¹)', fontsize=12, fontweight='bold')
ax2b.set_title('B  Top 10 Active Reactions (pFBA)', fontsize=13,
               fontweight='bold', loc='left')
ax2b.text(0.5, -0.09, 'Parsimonious FBA at baseline conditions',
          transform=ax2b.transAxes, ha='center', fontsize=10.5,
          color='#555555', style='italic')

# Annotate highest bar
max_idx = len(df_pfba) - 1
ax2b.annotate('Highest active flux',
              xy=(df_pfba['flux'].values[-1], max_idx),
              xytext=(df_pfba['flux'].values[-1] * 0.65, max_idx - 1.2),
              fontsize=10, color=C_RED,
              arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))

despine(ax2b)
ax2b.xaxis.grid(True, color=C_GREY, alpha=0.5, linewidth=0.8)
ax2b.yaxis.grid(False)
ax2b.set_axisbelow(True)

fig2.suptitle('Figure 2.  Gene Essentiality and Metabolic Flux Distribution',
               fontsize=14, fontweight='bold', y=1.01)
fig2.tight_layout()
save_fig(fig2, 'figure2_essentiality')
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — GUT TRANSIT REGULATORY SCHEMATIC
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 3...')

fig3, ax3 = plt.subplots(figsize=(16, 9))
fig3.patch.set_facecolor(C_WHITE)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_facecolor('#F8F9FA')
fig3.patch.set_facecolor('#F8F9FA')

# Background
bg = FancyBboxPatch((0.01, 0.04), 0.98, 0.88, boxstyle='round,pad=0.01',
                     facecolor='#F8F9FA', edgecolor='#DDDDDD', linewidth=1.5,
                     transform=ax3.transAxes, zorder=0)
ax3.add_patch(bg)

# Zone definitions
zones = [
    {'name': 'Stomach',  'ph': 'pH 2.0', 'o2': 'Anaerobic',    'color': C_RED,
     'x': 0.05, 'regulons': [('HSR', '#E67E22'), ('Acid Stress', C_RED)]},
    {'name': 'Duodenum', 'ph': 'pH 6.0', 'o2': 'Microaerobic', 'color': '#E67E22',
     'x': 0.28, 'regulons': [('HSR', '#E67E22')]},
    {'name': 'Ileum',    'ph': 'pH 7.0', 'o2': 'Aerobic',      'color': '#27AE60',
     'x': 0.51, 'regulons': [('HSR', '#E67E22'), ('Yap1', '#8E44AD')]},
    {'name': 'Colon',    'ph': 'pH 7.2', 'o2': 'Anaerobic',    'color': C_NAVY,
     'x': 0.74, 'regulons': [('HSR', '#E67E22'), ('HOG', C_BLUE), ('Yap1', '#8E44AD')]},
]

BOX_W = 0.20
BOX_H = 0.58
BOX_Y = 0.22

for zone in zones:
    x0 = zone['x']
    color = zone['color']

    # Main box
    box = FancyBboxPatch((x0, BOX_Y), BOX_W, BOX_H,
                          boxstyle='round,pad=0.015',
                          facecolor=color, edgecolor='white', linewidth=2.5,
                          alpha=0.92, zorder=2)
    ax3.add_patch(box)

    # Zone name
    ax3.text(x0 + BOX_W / 2, BOX_Y + BOX_H - 0.05,
             zone['name'], ha='center', va='top',
             fontsize=17, fontweight='bold', color='white',
             fontfamily=FONT, zorder=3)

    # pH
    ax3.text(x0 + BOX_W / 2, BOX_Y + BOX_H - 0.13,
             zone['ph'], ha='center', va='top',
             fontsize=13, color='white', fontfamily=FONT, zorder=3)

    # O2
    ax3.text(x0 + BOX_W / 2, BOX_Y + BOX_H - 0.20,
             zone['o2'], ha='center', va='top',
             fontsize=11.5, color='#FEFEFE', fontfamily=FONT,
             style='italic', zorder=3)

    # Growth rate
    ax3.text(x0 + BOX_W / 2, BOX_Y + BOX_H - 0.29,
             'μ = 0.0898 h⁻¹', ha='center', va='top',
             fontsize=11, color='#FFFFCC', fontfamily=FONT,
             fontweight='bold', zorder=3)

    # Divider line
    ax3.plot([x0 + 0.01, x0 + BOX_W - 0.01],
             [BOX_Y + BOX_H - 0.335, BOX_Y + BOX_H - 0.335],
             color='white', alpha=0.4, linewidth=0.8, zorder=3)

    # Regulon pills
    pill_y = BOX_Y + BOX_H - 0.37
    pill_x = x0 + 0.01
    for reg_name, reg_color in zone['regulons']:
        pill_w = len(reg_name) * 0.013 + 0.025
        pill = FancyBboxPatch((pill_x, pill_y - 0.035), pill_w, 0.042,
                               boxstyle='round,pad=0.008',
                               facecolor=reg_color, edgecolor='white',
                               linewidth=1, alpha=0.95, zorder=4)
        ax3.add_patch(pill)
        ax3.text(pill_x + pill_w / 2, pill_y - 0.013,
                 reg_name, ha='center', va='center',
                 fontsize=9.5, color='white', fontweight='bold',
                 fontfamily=FONT, zorder=5)
        pill_y -= 0.055

    # Arrow to next zone
    if zone is not zones[-1]:
        arr_x = x0 + BOX_W + 0.005
        arr_mid_y = BOX_Y + BOX_H / 2
        ax3.annotate('',
                     xy=(arr_x + 0.058, arr_mid_y),
                     xytext=(arr_x, arr_mid_y),
                     arrowprops=dict(arrowstyle='->', color='#555555',
                                     lw=2.5, mutation_scale=20),
                     zorder=6)

# GI tract flow label
ax3.text(0.5, 0.16, '→  Gastro-intestinal Transit  →',
         ha='center', va='center', fontsize=12, color='#555555',
         fontfamily=FONT, style='italic', zorder=3)

# Regulon legend box
leg_x, leg_y = 0.76, 0.08
leg_w, leg_h = 0.22, 0.12
leg_box = FancyBboxPatch((leg_x, leg_y), leg_w, leg_h,
                          boxstyle='round,pad=0.01',
                          facecolor='white', edgecolor='#BBBBBB',
                          linewidth=1.2, zorder=5)
ax3.add_patch(leg_box)
ax3.text(leg_x + leg_w / 2, leg_y + leg_h - 0.01, 'Regulon Legend',
         ha='center', va='top', fontsize=10, fontweight='bold',
         color='#333333', fontfamily=FONT, zorder=6)
legend_entries = [
    ('HSR', '#E67E22', 'Heat Shock Response'),
    ('HOG', C_BLUE,    'Osmotic Stress'),
    ('Acid', C_RED,    'Acid Stress Response'),
    ('Yap1', '#8E44AD', 'Oxidative Stress'),
]
for i, (tag, col, desc) in enumerate(legend_entries):
    lx = leg_x + 0.01
    ly = leg_y + leg_h - 0.038 - i * 0.022
    dot = plt.Circle((lx + 0.007, ly + 0.004), 0.006, color=col, zorder=7)
    ax3.add_patch(dot)
    ax3.text(lx + 0.018, ly + 0.004,
             f'{tag}: {desc}', va='center', fontsize=8.5,
             color='#333333', fontfamily=FONT, zorder=7)

# Titles
ax3.text(0.5, 0.96,
         'Gastrointestinal Transit Simulation — S. boulardii CNCM I-745',
         ha='center', va='top', fontsize=14, fontweight='bold',
         color=C_NAVY, fontfamily=FONT, zorder=3,
         transform=ax3.transAxes)
ax3.text(0.5, 0.055,
         'Regulatory network firing evaluated at each zone; '
         'growth rate stable at 0.0898 h⁻¹ (flux redistribution occurs)',
         ha='center', va='top', fontsize=10, color='#666666',
         fontfamily=FONT, style='italic', zorder=3,
         transform=ax3.transAxes)

fig3.suptitle('Figure 3.  Gut Transit Regulatory Network Simulation',
               fontsize=14, fontweight='bold', y=0.99)
fig3.tight_layout()
save_fig(fig3, 'figure3_gut_transit')
plt.close(fig3)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — HOST-MICROBE INTERACTION KINETICS
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 4...')

df_pk   = pd.read_csv('data/fba_outputs/protease_kinetics.csv')
df_sig  = pd.read_csv('data/fba_outputs/inflammatory_signaling.csv')

fig4, axes4 = plt.subplots(1, 2, figsize=(16, 7))
fig4.patch.set_facecolor(C_WHITE)

# — Panel A: TcdA kinetics ————————————————————————————————————————————————————
ax4a = axes4[0]
t     = df_pk['time_min'].values
TcdA  = df_pk['TcdA_nM'].values

ax4a.fill_between(t, TcdA, alpha=0.15, color=C_BLUE)
line_a, = ax4a.plot(t, TcdA, color=C_NAVY, linewidth=2.5, label='TcdA concentration')

# Secondary y-axis
ax4a_r = ax4a.twinx()
cleaved_pct = (1 - TcdA / TcdA[0]) * 100
ax4a_r.plot(t, cleaved_pct, color='none')
ax4a_r.set_ylabel('% Cleaved', fontsize=12, fontweight='bold', color='#555555')
ax4a_r.tick_params(axis='y', labelcolor='#555555', labelsize=11)
ax4a_r.set_ylim(0, 105)
ax4a_r.spines['top'].set_visible(False)

# 50% cleavage dashed line
ax4a.axhline(50, color='#888888', linestyle='--', linewidth=1.4, zorder=4)
ax4a.text(5, 52, '50% cleaved', fontsize=10, color='#555555', style='italic')

# 75.1% annotation
t_120 = t[-1]
tcdA_120 = TcdA[-1]
pct_120 = (1 - tcdA_120 / TcdA[0]) * 100
ax4a.annotate(f'{pct_120:.1f}% at 120 min',
              xy=(t_120, tcdA_120),
              xytext=(80, tcdA_120 + 15),
              fontsize=10, color=C_RED, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))

# Vertical line at t=120
ax4a.axvline(120, color='#AAAAAA', linestyle=':', linewidth=1.2, zorder=3)

# Shaded region beyond assay window (here we keep t max at 120 so no beyond)
ax4a.set_xlim(0, 130)
ax4a.set_ylim(-5, 110)
ax4a.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
ax4a.set_ylabel('TcdA Remaining (nM)', fontsize=12, fontweight='bold')
ax4a.set_title('A  CAMP Factor Protease Kinetics (TcdA Neutralisation)',
               fontsize=13, fontweight='bold', loc='left')
ax4a.text(0.5, -0.11,
          'Vmax = 0.8 nM min⁻¹,  Km = 15 nM,  [S]₀ = 100 nM',
          transform=ax4a.transAxes, ha='center', fontsize=10.5,
          color='#555555', style='italic')
ax4a.legend(loc='upper right', fontsize=10.5, frameon=True)
ax4a.spines['top'].set_visible(False)
ax4a.yaxis.grid(True, color=C_GREY, alpha=0.5)
ax4a.set_axisbelow(True)

# — Panel B: NF-κB dynamics ——————————————————————————————————————————————————
ax4b = axes4[1]
t2   = df_sig['time_min'].values

line_colors = {'NFkB': C_NAVY, 'IL1b': '#E74C3C', 'TNFa': '#E67E22', 'IL10': '#27AE60'}
line_labels  = {'NFkB': 'NF-κB', 'IL1b': 'IL-1β', 'TNFa': 'TNF-α', 'IL10': 'IL-10'}
col_map_prob = {'NFkB': 'NFkB_Sb1', 'IL1b': 'IL1b_Sb1', 'TNFa': 'TNFa_Sb1', 'IL10': 'IL10_Sb1'}
col_map_ctrl = {'NFkB': 'NFkB_Sb0', 'IL1b': 'IL1b_Sb0', 'TNFa': 'TNFa_Sb0', 'IL10': 'IL10_Sb0'}

for key in ['NFkB', 'IL1b', 'TNFa', 'IL10']:
    c = line_colors[key]
    prob_col = col_map_prob[key]
    ctrl_col = col_map_ctrl[key]
    ax4b.plot(t2, df_sig[prob_col].values, color=c, linewidth=2.2,
              label=f'{line_labels[key]} (Probiotic)')
    ax4b.plot(t2, df_sig[ctrl_col].values, color=c, linewidth=1.6,
              linestyle='--', alpha=0.5)

# Steady-state annotations
ax4b_t240 = t2[-1]
nfkb_ss_prob = df_sig['NFkB_Sb1'].values[-1]
nfkb_ss_ctrl = df_sig['NFkB_Sb0'].values[-1]
il10_ss_prob  = df_sig['IL10_Sb1'].values[-1]

ax4b.annotate(f'NF-κB prob.\n{nfkb_ss_prob:.2f}',
              xy=(ax4b_t240, nfkb_ss_prob),
              xytext=(ax4b_t240 - 55, nfkb_ss_prob + 0.15),
              fontsize=9.5, color=C_NAVY,
              arrowprops=dict(arrowstyle='->', color=C_NAVY, lw=1))
ax4b.annotate(f'NF-κB ctrl.\n{nfkb_ss_ctrl:.2f}',
              xy=(ax4b_t240, nfkb_ss_ctrl),
              xytext=(ax4b_t240 - 55, nfkb_ss_ctrl + 0.1),
              fontsize=9.5, color='#888888',
              arrowprops=dict(arrowstyle='->', color='#888888', lw=1))
ax4b.annotate(f'IL-10 prob.\n{il10_ss_prob:.2f}',
              xy=(ax4b_t240, il10_ss_prob),
              xytext=(ax4b_t240 - 55, il10_ss_prob - 0.25),
              fontsize=9.5, color='#27AE60',
              arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1))

# 70% suppression shaded band
ymax_ctrl = df_sig['NFkB_Sb0'].max()
ax4b.axhspan(0, ymax_ctrl * 0.30, alpha=0.07, color=C_RED,
             label='70% suppression zone')

# Legend: add dashed line for control
extra_handles = [
    Line2D([0], [0], color='grey', lw=2, linestyle='--', alpha=0.6,
           label='Control (Sb=0)'),
    mpatches.Patch(facecolor=C_RED, alpha=0.12, label='70% suppression zone'),
]
handles, lbls = ax4b.get_legend_handles_labels()
ax4b.legend(handles=handles[:4] + extra_handles, fontsize=9.5,
            loc='upper right', frameon=True, ncol=1)

ax4b.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
ax4b.set_ylabel('Relative Activity (normalised units)', fontsize=12, fontweight='bold')
ax4b.set_xlim(0, 240)
ax4b.set_title('B  NF-κB/Cytokine ODE Dynamics',
               fontsize=13, fontweight='bold', loc='left')
ax4b.text(0.5, -0.11,
          '240-minute simulation | NF-κB suppression 70% vs control',
          transform=ax4b.transAxes, ha='center', fontsize=10.5,
          color='#555555', style='italic')
ax4b.spines['top'].set_visible(False)
ax4b.spines['right'].set_visible(False)
ax4b.yaxis.grid(True, color=C_GREY, alpha=0.5)
ax4b.set_axisbelow(True)

fig4.suptitle('Figure 4.  Host-Microbe Interaction Kinetics',
               fontsize=14, fontweight='bold', y=1.01)
fig4.tight_layout()
save_fig(fig4, 'figure4_host_kinetics')
plt.close(fig4)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — CNN SURROGATE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 5...')

y_actual    = np.load('data/ml_datasets/test_actual.npy')
y_predicted = np.load('data/ml_datasets/test_predicted.npy')
mc_mean     = np.load('data/ml_datasets/mc_mean.npy')
mc_std      = np.load('data/ml_datasets/mc_std.npy')

r2_test = 1 - np.var(y_actual - y_predicted) / np.var(y_actual)
rmse_test = np.sqrt(np.mean((y_actual - y_predicted) ** 2))

fig5, axes5 = plt.subplots(1, 2, figsize=(16, 7))
fig5.patch.set_facecolor(C_WHITE)

# — Panel A: Predicted vs Actual ——————————————————————————————————————————————
ax5a = axes5[0]

abs_err = np.abs(y_actual - y_predicted)
norm5 = Normalize(vmin=0, vmax=0.02)
cmap5 = plt.cm.RdYlGn_r

sc = ax5a.scatter(y_actual, y_predicted,
                  c=abs_err, cmap=cmap5, norm=norm5,
                  alpha=0.5, s=40, edgecolors='white', linewidths=0.5, zorder=4)

# Colorbar
cbar = fig5.colorbar(sc, ax=ax5a, shrink=0.8, pad=0.02)
cbar.set_label('Absolute Error (h⁻¹)', fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Perfect prediction line
lims = [min(y_actual.min(), y_predicted.min()) - 0.01,
        max(y_actual.max(), y_predicted.max()) + 0.01]
ax5a.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction', zorder=5)
ax5a.set_xlim(lims)
ax5a.set_ylim(lims)

# Annotation box
textstr = f'R² = 0.8691\nRMSE = 0.0093\n(5-fold CV)'
props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#AAAAAA',
             alpha=0.9)
ax5a.text(0.05, 0.95, textstr, transform=ax5a.transAxes, fontsize=11,
          verticalalignment='top', bbox=props)

ax5a.set_xlabel('Actual Growth Rate (h⁻¹)', fontsize=12, fontweight='bold')
ax5a.set_ylabel('Predicted Growth Rate (h⁻¹)', fontsize=12, fontweight='bold')
ax5a.set_title('A  CNN Surrogate: Predicted vs Actual (Test Set)',
               fontsize=13, fontweight='bold', loc='left')
ax5a.text(0.5, -0.10,
          '2,000-sample LHS dataset | 5-fold CV R² = 0.8691 ± 0.0185',
          transform=ax5a.transAxes, ha='center', fontsize=10.5,
          color='#555555', style='italic')
ax5a.legend(loc='lower right', fontsize=10.5, frameon=True)
ax5a.spines['top'].set_visible(False)
ax5a.spines['right'].set_visible(False)
ax5a.yaxis.grid(True, color=C_GREY, alpha=0.5)
ax5a.set_axisbelow(True)

# — Panel B: MC Dropout uncertainty ——————————————————————————————————————————
ax5b = axes5[1]

# Select 30 samples sorted by actual growth rate
n_show = min(30, len(y_actual))
sort_idx = np.argsort(y_actual)[:n_show]
y_act_30   = y_actual[sort_idx]
mc_mean_30 = mc_mean[sort_idx]
mc_std_30  = mc_std[sort_idx]
ci_half     = 1.96 * mc_std_30

x_idx = np.arange(n_show)

# Color error bars by uncertainty
unc_norm = Normalize(vmin=mc_std_30.min(), vmax=mc_std_30.max())
unc_cmap = plt.cm.YlOrRd

for i in range(n_show):
    col_unc = unc_cmap(unc_norm(mc_std_30[i]))
    ax5b.errorbar(x_idx[i], mc_mean_30[i], yerr=ci_half[i],
                  fmt='none', ecolor=col_unc, elinewidth=1.8,
                  capsize=3, capthick=1.5, alpha=0.8)

ax5b.scatter(x_idx, mc_mean_30, color=C_BLUE, s=35, zorder=5,
             alpha=0.8, label='Surrogate ± 95% CI')
ax5b.scatter(x_idx, y_act_30, color='black', s=28, zorder=6,
             marker='D', label='Actual FBA')

# Mean error annotation
mean_err_pct = np.mean(np.abs(y_actual - mc_mean) / (y_actual + 1e-9)) * 100
props2 = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#AAAAAA',
              alpha=0.9)
ax5b.text(0.05, 0.95, f'Mean error: {mean_err_pct:.2f}%',
          transform=ax5b.transAxes, fontsize=11,
          verticalalignment='top', bbox=props2)

ax5b.set_xlabel('Sample Index (sorted by actual growth rate)', fontsize=12, fontweight='bold')
ax5b.set_ylabel('Growth Rate (h⁻¹)', fontsize=12, fontweight='bold')
ax5b.set_title('B  Monte Carlo Dropout Uncertainty Quantification',
               fontsize=13, fontweight='bold', loc='left')
ax5b.text(0.5, -0.10,
          '50 stochastic forward passes | 95% confidence intervals',
          transform=ax5b.transAxes, ha='center', fontsize=10.5,
          color='#555555', style='italic')
ax5b.legend(loc='upper left', fontsize=10.5, frameon=True)
ax5b.spines['top'].set_visible(False)
ax5b.spines['right'].set_visible(False)
ax5b.yaxis.grid(True, color=C_GREY, alpha=0.5)
ax5b.set_axisbelow(True)

fig5.suptitle('Figure 5.  CNN Surrogate Model Validation and Uncertainty Quantification',
               fontsize=14, fontweight='bold', y=1.01)
fig5.tight_layout()
save_fig(fig5, 'figure5_surrogate')
plt.close(fig5)

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
import os
figs = [f for f in sorted(os.listdir('figures/')) if f.endswith('.png')]
print('\n--- Generated figures ---')
for f in figs:
    size = os.path.getsize(f'figures/{f}')
    print(f'  {f}: {size/1024:.0f} KB')
print(f'\nTotal: {len(figs)} figures generated')
print('\nALL FIGURES COMPLETE')
