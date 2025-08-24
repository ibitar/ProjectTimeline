# app.py
# Streamlit Gantt Planner — Ibrahim's options
# Run: streamlit run app.py

import io
import math
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
from datetime import datetime, date

st.set_page_config(page_title="Gantt Planner", layout="wide")

# ============================== Helpers ==============================

def days_between(d0, d1, inclusive=True):
    """Return duration in days between two python dates."""
    delta = (d1 - d0).days
    return delta + 1 if inclusive else delta

def weeks_between(d0, d1, inclusive=True):
    d = days_between(d0, d1, inclusive=inclusive)
    return d / 7.0

def to_date(x):
    if pd.isna(x):
        return None
    if isinstance(x, date):
        return x
    if isinstance(x, datetime):
        return x.date()
    return pd.to_datetime(str(x)).date()

def _data_width_in_pixels(ax, x_left, x_right):
    p0 = ax.transData.transform((x_left, 0))
    p1 = ax.transData.transform((x_right, 0))
    return abs(p1[0] - p0[0])

def _px_to_data_y(ax, px):
    fig = ax.figure
    fig.canvas.draw()
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((0, 1))
    py_per_data = abs(p1[1] - p0[1])
    return 0.0 if py_per_data == 0 else px / py_per_data

def _fit_text_in_span(ax, text_str, x_center, y, x_left, x_right,
                      max_font_size=10, min_font_size=6, padding_px=6, zorder=6,
                      ha="center", va="center", clip_on=False):
    """Place text at (x_center, y) and shrink font to fit within [x_left, x_right]."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    available_px = max(_data_width_in_pixels(ax, x_left, x_right) - padding_px, 0)

    for fs in range(max_font_size, min_font_size - 1, -1):
        txt = ax.text(x_center, y, text_str, va=va, ha=ha,
                      fontsize=fs, clip_on=clip_on, zorder=zorder)
        bbox = txt.get_window_extent(renderer=renderer)
        if bbox.width <= available_px:
            return fs, txt
        txt.remove()
    return None, None

def demo_dataframe():
    df = pd.DataFrame([
        ["T1", "Cadrage du projet", "2025-09-01", "2025-09-05", "Pilotage", 0, ""],
        ["T2", "Spécifications fonctionnelles", "2025-09-08", "2025-09-19", "Conception", 0, "T1"],
        ["T3", "Architecture technique", "2025-09-15", "2025-09-26", "Conception", 0, "T1"],
        ["M1", "Milestone: Specs validées", "2025-09-22", "2025-09-22", "Jalons", 1, "T2,T3"],
        ["T4", "Développement sprint 1", "2025-09-23", "2025-10-10", "Dév", 0, "M1"],
        ["T5", "Développement sprint 2", "2025-10-13", "2025-10-31", "Dév", 0, "T4"],
        ["T6", "Tests & Recette", "2025-11-03", "2025-11-14", "Qualité", 0, "T5"],
        ["M2", "Milestone: Go/No-Go", "2025-11-17", "2025-11-17", "Jalons", 1, "T6"],
    ], columns=["id", "title", "start", "end", "group", "milestone", "depends_on"])
    return df

def parse_combined_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["id","title","start","end"]
    for r in required:
        if r not in cols:
            raise ValueError("CSV combiné attendu avec colonnes au minimum: id,title,start,end[,group,milestone,depends_on]")
    # harmonize
    df = df.rename(columns={cols.get("id","id"): "id",
                            cols.get("title","title"): "title",
                            cols.get("start","start"): "start",
                            cols.get("end","end"): "end",
                            cols.get("group","group"): "group",
                            cols.get("milestone","milestone"): "milestone",
                            cols.get("depends_on","depends_on"): "depends_on"})
    if "group" not in df: df["group"] = ""
    if "milestone" not in df: df["milestone"] = 0
    if "depends_on" not in df: df["depends_on"] = ""
    return df

def parse_activities_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    needed = {"id","title","start","end"}
    if not needed.issubset({c.lower() for c in df.columns}):
        raise ValueError("activities.csv doit contenir: id,title,start,end[,group,depends_on]")
    return df

def parse_milestones_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # accept either (id,title,date) or (id,title,start,end with same day)
    if {"id","title","date"}.issubset({c.lower() for c in df.columns}):
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df["start"] = df["date"]
        df["end"] = df["date"]
    elif {"id","title","start","end"}.issubset({c.lower() for c in df.columns}):
        pass
    else:
        raise ValueError("milestones.csv doit contenir: id,title,date OR id,title,start,end (start=end).")
    df["milestone"] = 1
    return df

def parse_inputs_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # columns: date,label
    if not {"date","label"}.issubset({c.lower() for c in df.columns}):
        raise ValueError("inputs.csv doit contenir: date,label")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df[["date","label"]]

# ============================== Sidebar ==============================

st.sidebar.header("⚙️ Options")

st.sidebar.subheader("Données")
mode_data = st.sidebar.radio("Source des données", ["CSV combiné", "CSV séparés", "Démo"], horizontal=True)

uploaded_combined = None
uploaded_acts = None
uploaded_miles = None
uploaded_inputs = None

if mode_data == "CSV combiné":
    uploaded_combined = st.sidebar.file_uploader("Importer le CSV combiné (id,title,start,end[,group,milestone,depends_on])", type=["csv"])
    uploaded_inputs = st.sidebar.file_uploader("Importer inputs.csv (date,label) — optionnel", type=["csv"])
elif mode_data == "CSV séparés":
    uploaded_acts = st.sidebar.file_uploader("activities.csv (id,title,start,end[,group,depends_on])", type=["csv"])
    uploaded_miles = st.sidebar.file_uploader("milestones.csv (id,title,date) ou (id,title,start,end)", type=["csv"])
    uploaded_inputs = st.sidebar.file_uploader("inputs.csv (date,label) — optionnel", type=["csv"])

st.sidebar.markdown("---")
unit = st.sidebar.selectbox("Unité de temps", ["Jours", "Semaines"])
inclusive_duration = st.sidebar.checkbox("Durée inclusive (inclure le jour de fin)", value=True)

st.sidebar.subheader("Axes & Grille")
rot = st.sidebar.slider("Rotation des dates (X)", 0, 90, 30, step=5)
show_grid = st.sidebar.checkbox("Quadrillage pointillé", value=True)
grid_axis = st.sidebar.selectbox("Grille sur", ["x", "both"], index=0)
x_tick_step = st.sidebar.number_input("Pas des graduations majeures (jours/semaines)", min_value=1, max_value=30, value=1, step=1)
date_fmt = st.sidebar.text_input("Format des dates (strftime)", "%d %b %Y")

st.sidebar.subheader("Plage X")
x_min_str = st.sidebar.text_input("X min (AAAA-MM-JJ) — optionnel", "")
x_max_str = st.sidebar.text_input("X max (AAAA-MM-JJ) — optionnel", "")
x_margin_ratio = st.sidebar.slider("Marge aux extrémités (%)", 0, 20, 5, step=1) / 100.0

st.sidebar.subheader("Visuel")
row_bg = st.sidebar.checkbox("Bandes horizontales de fond", value=True)
row_bg_alpha = st.sidebar.slider("Transparence bandes", 0.0, 0.5, 0.10, step=0.01)
row_bg_height = st.sidebar.slider("Épaisseur bande (0–1)", 0.1, 1.0, 0.6, step=0.05)
bar_height = st.sidebar.slider("Hauteur des barres", 0.1, 1.0, 0.4, step=0.05)

st.sidebar.subheader("Titres / Durées")
titles_above = st.sidebar.checkbox("Titres au-dessus des barres", value=True)
title_max_fs = st.sidebar.slider("Taille max titres", 6, 18, 10)
title_min_fs = st.sidebar.slider("Taille min titres", 4, 12, 6)
title_gap_px = st.sidebar.slider("Décalage vertical titres (px)", 0, 20, 6)
title_padding_px = st.sidebar.slider("Marge interne titres (px)", 0, 20, 6)

dur_in_bar = st.sidebar.checkbox("Afficher la durée dans la barre", value=True)
dur_font = st.sidebar.slider("Taille police durée", 6, 14, 8)
dur_fmt_days = st.sidebar.text_input("Format durée (jours)", "{d} j")
dur_fmt_weeks = st.sidebar.text_input("Format durée (semaines)", "{w:.1f} sem")

st.sidebar.subheader("Dates & End-caps")
show_start_end = st.sidebar.checkbox("Étiquettes date début/fin", value=True)
date_label_offset = st.sidebar.number_input("Décalage étiquettes (jours)", value=0.2, step=0.1, format="%.1f")
draw_endcaps = st.sidebar.checkbox("Traits verticaux aux extrémités", value=True)
endcap_len = st.sidebar.slider("Longueur end-caps (en Y)", 0.1, 1.0, 0.35, step=0.05)

st.sidebar.subheader("Jalons")
milestones_vlines = st.sidebar.checkbox("Lignes verticales jalons", value=True)
ms_markersize = st.sidebar.slider("Taille triangles jalons", 8, 36, 16, step=1)
ms_offset = st.sidebar.slider("Offset vertical jalons (axes fraction)", 0.0, 0.1, 0.02, step=0.005)
anti_overlap = st.sidebar.checkbox("Anti-chevauchement (titres/jalons/inputs)", value=True)
ms_alt_extra = st.sidebar.slider("Alternance offset jalons (ajout max)", 0.0, 0.08, 0.03, step=0.005)

st.sidebar.subheader("Inputs (flèches descendantes)")
show_inputs = st.sidebar.checkbox("Afficher inputs", value=True)
inputs_alt_extra = st.sidebar.slider("Alternance hauteur inputs (ajout max)", 0.0, 1.0, 0.4, step=0.05)
inputs_top_margin = st.sidebar.slider("Marge verticale haut (Y)", 0.0, 2.0, 1.0, step=0.1)
input_arrow_len = st.sidebar.slider("Longueur flèche input (Y)", 0.2, 2.0, 0.8, step=0.1)

st.sidebar.subheader("Axe X (flèche)")
ax_arrow_lw = st.sidebar.slider("Épaisseur flèche axe X", 2.0, 16.0, 8.0, step=0.5)
ax_arrow_scale = st.sidebar.slider("Taille pointe flèche", 10, 60, 30, step=2)

st.sidebar.markdown("---")
fig_w = st.sidebar.slider("Largeur figure", 6, 20, 12)
fig_h = st.sidebar.slider("Hauteur figure", 4, 12, 6)
st.sidebar.caption("💡 Télécharge le PNG en bas de page.")

# ============================== Data loading ==============================

try:
    if mode_data == "Démo":
        df_all = demo_dataframe()
        df_inputs = pd.DataFrame({"date": ["2025-09-10", "2025-10-05"], "label": ["Données A1 - EDV", "Données A2 - EDV"]})
    elif mode_data == "CSV combiné":
        if uploaded_combined is None:
            st.info("➡️ Importez un CSV combiné ou passez en mode Démo.")
            st.stop()
        df_all = parse_combined_csv(uploaded_combined)
        df_inputs = parse_inputs_csv(uploaded_inputs) if uploaded_inputs else pd.DataFrame(columns=["date","label"])
    else:  # CSV séparés
        if uploaded_acts is None or uploaded_miles is None:
            st.info("➡️ Importez activities.csv et milestones.csv, ou passez en mode Démo.")
            st.stop()
        df_act = parse_activities_csv(uploaded_acts)
        df_ms  = parse_milestones_csv(uploaded_miles)
        # Normalize
        df_act = df_act.rename(columns={c: c.lower() for c in df_act.columns})
        df_ms  = df_ms.rename(columns={c: c.lower() for c in df_ms.columns})
        if "group" not in df_act: df_act["group"] = ""
        if "depends_on" not in df_act: df_act["depends_on"] = ""
        df_act["milestone"] = 0
        df_all = pd.concat([df_act[["id","title","start","end","group","milestone","depends_on"]],
                            df_ms[["id","title","start","end","milestone"]].assign(group="", depends_on="")],
                           ignore_index=True)
        df_inputs = parse_inputs_csv(uploaded_inputs) if uploaded_inputs else pd.DataFrame(columns=["date","label"])
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()

# --- Validation basique des colonnes start/end ---
if "start" not in df_all or "end" not in df_all:
    st.error("Le CSV ne contient pas de colonnes start/end valides.")
    st.stop()

if df_all[["start","end"]].isna().any().any():
    st.error("Certaines lignes ont des dates start/end manquantes ou invalides. Corrige ton CSV.")
    st.stop()
    
# Parse dates
# --- Initialisation de df_all pour éviter NameError ---
if "df_all" not in locals():
    df_all = pd.DataFrame(columns=["id","title","start","end","group","milestone","depends_on"])
    df_inputs = pd.DataFrame(columns=["date","label"])

# Parse dates
df_all["start"] = pd.to_datetime(df_all["start"]).dt.date
df_all["end"]   = pd.to_datetime(df_all["end"]).dt.date
df_all = df_all.sort_values("start").reset_index(drop=True)
df_ms = df_all[(df_all.get("milestone", 0).astype(int) == 1) | (df_all["start"] == df_all["end"])].copy()
df_tasks = df_all[~df_all.index.isin(df_ms.index)].copy()

if not df_inputs.empty:
    df_inputs["date"] = pd.to_datetime(df_inputs["date"]).dt.date

# ============================== X range (robuste) ==============================

def _first_date_or_none(series):
    try:
        if series is None or len(series) == 0:
            return None
        s = pd.to_datetime(series, errors="coerce").dropna()
        return s.min().date() if not s.empty else None
    except Exception:
        return None

def _last_date_or_none(series):
    try:
        if series is None or len(series) == 0:
            return None
        s = pd.to_datetime(series, errors="coerce").dropna()
        return s.max().date() if not s.empty else None
    except Exception:
        return None

# Si aucune donnée, avertir et stopper proprement
if df_all.empty and (df_inputs.empty if 'df_inputs' in locals() else True):
    st.warning("Aucune donnée valide chargée. Importez un CSV ou passez en mode Démo.")
    st.stop()

# bornes issues des données (tâches + jalons + inputs si présents)
data_min = _first_date_or_none(df_all["start"]) or _first_date_or_none(df_all["end"])
data_max = _last_date_or_none(df_all["end"]) or _last_date_or_none(df_all["start"])

if 'df_inputs' in locals() and not df_inputs.empty:
    in_min = _first_date_or_none(df_inputs["date"])
    in_max = _last_date_or_none(df_inputs["date"])
    if in_min:
        data_min = in_min if (data_min is None or in_min < data_min) else data_min
    if in_max:
        data_max = in_max if (data_max is None or in_max > data_max) else data_max

# fallback si toujours None → fenêtre centrée sur aujourd’hui
if data_min is None or data_max is None:
    today = pd.Timestamp.today().date()
    data_min = today
    data_max = today

# overrides utilisateur
def _parse_date_or(default, s):
    try:
        return pd.to_datetime(s).date() if s else default
    except Exception:
        return default

x_min_user = _parse_date_or(data_min, x_min_str)
x_max_user = _parse_date_or(data_max, x_max_str)

# corriger inversion éventuelle
if x_min_user > x_max_user:
    x_min_user, x_max_user = x_max_user, x_min_user

# conversion en nombres Matplotlib
x_min_num_raw = mdates.date2num(x_min_user)
x_max_num_raw = mdates.date2num(x_max_user)

# éviter plage nulle ou négative
if not np.isfinite(x_min_num_raw) or not np.isfinite(x_max_num_raw):
    st.warning("Impossible de déterminer la plage des dates (NaN/Inf). Vérifie tes colonnes de dates.")
    st.stop()

if x_max_num_raw <= x_min_num_raw:
    x_max_num_raw = x_min_num_raw + 1.0  # +1 jour

# marge
total_range = x_max_num_raw - x_min_num_raw
margin = max(0.0, total_range * x_margin_ratio)
x_min_num = x_min_num_raw - margin
x_max_num = x_max_num_raw + margin

# ============================== Draw ==============================

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# Y layout
y_positions = range(1, len(df_tasks) + 1)
ax.set_yticks(list(y_positions))
ax.set_yticklabels([])
ax.tick_params(axis='y', left=False, labelleft=False)

# Background bands
if row_bg:
    for i in y_positions:
        y0 = i - row_bg_height / 2.0
        y1 = i + row_bg_height / 2.0
        ax.axhspan(y0, y1, xmin=0.0, xmax=1.0, alpha=row_bg_alpha, zorder=1)

# Bars + labels
for i, row in enumerate(df_tasks.itertuples(index=False), start=1):
    start_num = mdates.date2num(row.start)
    end_num   = mdates.date2num(row.end)
    duration_num = end_num - start_num
    duration_days = days_between(row.start, row.end, inclusive=inclusive_duration)
    duration_weeks = duration_days / 7.0

    # bar
    ax.barh(i, duration_num, left=start_num, height=bar_height, align='center', zorder=2)

    # title above (with anti-overlap by alternating height)
    if titles_above:
        add_y = _px_to_data_y(ax, title_gap_px)
        alt = (i % 2) * _px_to_data_y(ax, 6) if anti_overlap else 0.0
        y_title = i + (row_bg_height / 2.0) + add_y + alt
        _fit_text_in_span(
            ax, row.title,
            x_center=start_num + duration_num / 2.0,
            y=y_title, x_left=start_num, x_right=end_num,
            max_font_size=title_max_fs, min_font_size=title_min_fs, padding_px=title_padding_px,
            zorder=6, ha="center", va="center", clip_on=False
        )

    # duration inside bar (unit-aware)
    if dur_in_bar and duration_num > 0:
        dur_txt = dur_fmt_days.format(d=duration_days) if unit == "Jours" else dur_fmt_weeks.format(w=duration_weeks)
        ax.text(start_num + duration_num/2.0, i, dur_txt, va="center", ha="center", fontsize=dur_font, zorder=5, clip_on=True)

    # start/end labels + endcaps
    if show_start_end:
        left  = mdates.num2date(start_num).strftime("%d/%m")
        right = mdates.num2date(end_num).strftime("%d/%m")
        # alternate tiny vertical jitter to reduce overlaps
        jitter = (0.15 if (anti_overlap and i % 2 == 0) else 0.0)
        ax.text(start_num - date_label_offset, i + jitter, left,  va="center", ha="right", fontsize=9, zorder=5, clip_on=False)
        ax.text(end_num + date_label_offset,   i - jitter, right, va="center", ha="left",  fontsize=9, zorder=5, clip_on=False)
    if draw_endcaps:
        ax.vlines([start_num, end_num], ymin=i - endcap_len, ymax=i + endcap_len, zorder=5)

# Milestones at y=0
palette = plt.get_cmap("tab10").colors
blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
legend_handles = []

for k, row in enumerate(df_ms.itertuples(index=False)):
    x_ms = mdates.date2num(row.start)
    # vline
    if milestones_vlines:
        ax.axvline(x=x_ms, linestyle=":", alpha=0.3, zorder=3)
    # alternating vertical offset (axes fraction)
    extra = (ms_alt_extra if (anti_overlap and k % 2 == 1) else 0.0)
    y_af = ms_offset + extra
    color = palette[k % len(palette)]
    ax.plot([x_ms], [y_af], marker="v", markersize=ms_markersize,
            markerfacecolor=color, markeredgecolor=color,
            transform=blended, zorder=7)
    legend_handles.append(Line2D([0],[0], marker="v", linestyle="None",
                                 markersize=ms_markersize,
                                 markerfacecolor=color, markeredgecolor=color,
                                 label=row.title))

if legend_handles:
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)

# Inputs (down arrows at top)
if show_inputs and not df_inputs.empty:
    y_top = len(df_tasks) + inputs_top_margin
    for k, row in enumerate(df_inputs.itertuples(index=False)):
        x_d = mdates.date2num(row.date)
        extra = (inputs_alt_extra if (anti_overlap and k % 2 == 1) else 0.0)
        y0 = y_top + extra
        y1 = y0 - input_arrow_len
        # arrow
        ax.annotate("", xy=(x_d, y1), xytext=(x_d, y0), xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle='-|>', lw=1.5), zorder=6)
        # vertical label near arrow head, rotated
        ax.text(x_d, y1 - 0.05, row.label, rotation=90, va="top", ha="center", fontsize=9, zorder=6, clip_on=False)

# X axis limits and ticks
ax.set_xlim(x_min_num, x_max_num)
ax.set_ylim(-0.5, max(2.0, len(df_tasks) + inputs_top_margin + 0.8))
ax.xaxis_date()
if unit == "Jours":
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=int(x_tick_step)))
else:
    # step in weeks
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=int(x_tick_step)))
ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")

# Grid
if show_grid:
    ax.grid(True, axis=grid_axis, linestyle=":", alpha=0.6)

# Remove frame
for sp in ax.spines.values():
    sp.set_visible(False)

# X-axis arrow (bottom)
ax.annotate(
    '',
    xy=(x_max_num, 0.0), xycoords=('data', 'axes fraction'),
    xytext=(x_min_num, 0.0), textcoords=('data', 'axes fraction'),
    arrowprops=dict(arrowstyle='-|>', color='red', lw=ax_arrow_lw,
                    shrinkA=0, shrinkB=0, mutation_scale=ax_arrow_scale),
    zorder=6
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Planning")

st.pyplot(fig, use_container_width=True)

# ============================== Download PNG ==============================

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
st.download_button(
    label="⬇️ Télécharger le PNG",
    data=buf.getvalue(),
    file_name="planning.png",
    mime="image/png",
)

# ============================== Help / Schemas ==============================

