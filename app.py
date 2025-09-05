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
        ["T1", "Cadrage du projet", "2025-09-01", "2025-09-05", "Pilotage", 0, 20, ""],
        ["T2", "Spécifications fonctionnelles", "2025-09-08", "2025-09-19", "Conception", 0, 0, "T1"],
        ["T3", "Architecture technique", "2025-09-15", "2025-09-26", "Conception", 0, 0, "T1"],
        ["M1", "Milestone: Specs validées", "2025-09-22", "2025-09-22", "Jalons", 1, 0, "T2,T3"],
        ["T4", "Développement sprint 1", "2025-09-23", "2025-10-10", "Dév", 0, 0, "M1"],
        ["T5", "Développement sprint 2", "2025-10-13", "2025-10-31", "Dév", 0, 0, "T4"],
        ["T6", "Tests & Recette", "2025-11-03", "2025-11-14", "Qualité", 0, 0, "T5"],
        ["M2", "Milestone: Go/No-Go", "2025-11-17", "2025-11-17", "Jalons", 1, 0, "T6"],
    ], columns=["id", "title", "start", "end", "group", "milestone", "progress", "depends_on"])
    return df

def parse_combined_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["id","title","start","end"]
    for r in required:
        if r not in cols:
            raise ValueError("CSV combiné attendu avec colonnes au minimum: id,title,start,end[,group,milestone,progress,depends_on]")
    # harmonize
    df = df.rename(columns={cols.get("id","id"): "id",
                            cols.get("title","title"): "title",
                            cols.get("start","start"): "start",
                            cols.get("end","end"): "end",
                            cols.get("group","group"): "group",
                            cols.get("milestone","milestone"): "milestone",
                            cols.get("progress","progress"): "progress",
                            cols.get("depends_on","depends_on"): "depends_on",
                            cols.get("marker","marker"): "marker"})
    if "group" not in df:
        df["group"] = ""
    if "milestone" not in df:
        df["milestone"] = 0
    if "progress" not in df:
        df["progress"] = 0
    if "depends_on" not in df:
        df["depends_on"] = ""
    if "marker" not in df:
        df["marker"] = ""

    # ensure text columns are strings for editing compatibility
    df["id"] = df["id"].astype(str)
    df["title"] = df["title"].astype(str)
    df["group"] = df["group"].fillna("").astype(str)
    df["depends_on"] = df["depends_on"].fillna("").astype(str)
    df["marker"] = df["marker"].fillna("").astype(str)
    df["progress"] = pd.to_numeric(df["progress"], errors="coerce").fillna(0)

    return df

def parse_activities_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})
    needed = {"id","title","start","end"}
    if not needed.issubset(set(df.columns)):
        raise ValueError("activities.csv doit contenir: id,title,start,end[,group,progress,depends_on]")
    if "progress" not in df:
        df["progress"] = 0
    df["progress"] = pd.to_numeric(df["progress"], errors="coerce").fillna(0)
    return df

def parse_milestones_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # accept either (id,title,date) or (id,title,start,end with same day)
    if {"id", "title", "date"}.issubset(df.columns):
        df["start"] = df["date"]
        df["end"] = df["date"]
    elif {"id", "title", "start", "end"}.issubset(df.columns):
        pass
    else:
        raise ValueError(
            "milestones.csv doit contenir: id,title,date OR id,title,start,end (start=end)."
        )
    if "marker" not in df:
        df["marker"] = "v"
    df["marker"] = df["marker"].replace("", "v").fillna("v").astype(str)
    df["milestone"] = 1
    return df

def parse_inputs_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # columns: date,label
    if not {"date","label"}.issubset({c.lower() for c in df.columns}):
        raise ValueError("inputs.csv doit contenir: date,label")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df[["date","label"]]

import datetime

def _init_editor_state():
    if "editor_tasks" not in st.session_state:
        st.session_state.editor_tasks = pd.DataFrame([
            {"id": "T1", "title": "Cadrage",
             "start": datetime.date(2025, 9, 1),
             "end": datetime.date(2025, 9, 5),
             "group": "Pilotage", "progress": 0, "depends_on": ""},
            {"id": "T2", "title": "Spécifications",
             "start": datetime.date(2025, 9, 8),
             "end": datetime.date(2025, 9, 19),
             "group": "Conception", "progress": 0, "depends_on": "T1"},
        ], columns=["id", "title", "start", "end", "group", "progress", "depends_on"])

    if "editor_ms" not in st.session_state:
        st.session_state.editor_ms = pd.DataFrame([
            {
                "id": "M1",
                "title": "Specs validées",
                "date": datetime.date(2025, 9, 22),
                "marker": "v",
            },
        ], columns=["id", "title", "date", "marker"])

    if "editor_inputs" not in st.session_state:
        st.session_state.editor_inputs = pd.DataFrame([
            {"date": datetime.date(2025, 9, 10), "label": "Input A"},
        ], columns=["date", "label"])

def _init_editor_state_from_df(df_all: pd.DataFrame, df_inputs: pd.DataFrame, file_id: str = ""):
    """Initialise the editor session_state from an uploaded CSV."""
    tasks = df_all[df_all.get("milestone", 0).astype(int) != 1].copy()
    tasks["progress"] = pd.to_numeric(tasks.get("progress", 0), errors="coerce").fillna(0)
    tasks = tasks[["id", "title", "start", "end", "group", "progress", "depends_on"]]
    tasks["id"] = tasks["id"].astype(str)
    tasks["title"] = tasks["title"].astype(str)
    tasks["group"] = tasks["group"].fillna("").astype(str)
    tasks["depends_on"] = tasks["depends_on"].fillna("").astype(str)

    tasks["start"] = pd.to_datetime(tasks["start"]).dt.date
    tasks["end"] = pd.to_datetime(tasks["end"]).dt.date

    ms = df_all[df_all.get("milestone", 0).astype(int) == 1].copy()
    ms = ms.rename(columns={"start": "date"})
    if "marker" not in ms:
        ms["marker"] = "v"
    ms = ms[["id", "title", "date", "marker"]]

    ms["id"] = ms["id"].astype(str)
    ms["title"] = ms["title"].astype(str)
    ms["marker"] = ms["marker"].replace("", "v").fillna("v").astype(str)

    ms["date"] = pd.to_datetime(ms["date"]).dt.date

    st.session_state.editor_tasks = tasks
    st.session_state.editor_ms = ms
    st.session_state.editor_inputs = df_inputs.copy()
    if "date" in st.session_state.editor_inputs:
        st.session_state.editor_inputs["date"] = pd.to_datetime(
            st.session_state.editor_inputs["date"]
        ).dt.date
    st.session_state.editor_source = "uploaded"
    st.session_state.editor_file = file_id

def _validate_editors(df_tasks: pd.DataFrame, df_ms: pd.DataFrame):
    errors = []
    # Tâches : colonnes requises
    req_t = {"id","title","start","end"}
    if not req_t.issubset(set(map(str.lower, df_tasks.columns))):
        errors.append("Activités: colonnes requises manquantes (id,title,start,end).")
    # Jalons : colonnes requises
    req_m = {"id","title","date"}
    if not req_m.issubset(set(map(str.lower, df_ms.columns))):
        errors.append("Jalons: colonnes requises manquantes (id,title,date).")

    # Cast dates
    for col in ["start","end"]:
        if col in df_tasks:
            df_tasks[col] = pd.to_datetime(df_tasks[col], errors="coerce").dt.date
    if "date" in df_ms:
        df_ms["date"] = pd.to_datetime(df_ms["date"], errors="coerce").dt.date
    if "progress" in df_tasks:
        df_tasks["progress"] = pd.to_numeric(df_tasks["progress"], errors="coerce")

    # Lignes invalides
    if df_tasks[["start","end"]].isna().any().any():
        errors.append("Activités: certaines dates start/end sont invalides.")
    if "progress" in df_tasks and df_tasks["progress"].isna().any():
        errors.append("Activités: certaines valeurs de progress sont invalides.")
    if "progress" in df_tasks and ((df_tasks["progress"] < 0) | (df_tasks["progress"] > 100)).any():
        errors.append("Activités: progress doit être entre 0 et 100.")
    # start <= end
    bad = df_tasks[(df_tasks["start"] > df_tasks["end"])]
    if not bad.empty:
        errors.append(f"Activités: {len(bad)} ligne(s) avec start > end.")
    # jalon (date non nulle)
    if df_ms["date"].isna().any():
        errors.append("Jalons: certaines dates sont invalides.")
    # IDs uniques (sur l’ensemble)
    ids = list(df_tasks.get("id",[])) + list(df_ms.get("id",[]))
    if len(ids) != len(set(ids)):
        errors.append("IDs dupliqués entre activités/jalons.")

    return errors

def _build_df_all_from_editors(df_tasks: pd.DataFrame, df_ms: pd.DataFrame) -> pd.DataFrame:
    # Normalise et fusionne
    df_tasks = df_tasks.copy()
    df_ms = df_ms.copy()
    df_tasks["milestone"] = 0
    df_tasks["group"] = df_tasks.get("group", "")
    df_tasks["group"] = df_tasks["group"].fillna("").astype(str)
    df_tasks["depends_on"] = df_tasks.get("depends_on", "")
    df_tasks["depends_on"] = df_tasks["depends_on"].fillna("").astype(str)
    df_tasks["marker"] = df_tasks.get("marker", "")
    df_tasks["marker"] = df_tasks["marker"].fillna("").astype(str)
    df_tasks["progress"] = pd.to_numeric(df_tasks.get("progress", 0), errors="coerce").fillna(0)
    df_ms = df_ms.rename(columns={"date": "start"})
    df_ms["end"] = df_ms["start"]
    df_ms["group"] = ""
    df_ms["depends_on"] = ""
    df_ms["milestone"] = 1
    df_ms["marker"] = df_ms.get("marker", "v")
    df_ms["marker"] = df_ms["marker"].replace("", "v").fillna("v").astype(str)
    df_ms["progress"] = 0
    df_ms = df_ms[["id", "title", "start", "end", "group", "milestone", "progress", "depends_on", "marker"]]
    df_tasks = df_tasks[["id", "title", "start", "end", "group", "milestone", "progress", "depends_on", "marker"]]
    df_all = pd.concat([df_tasks, df_ms], ignore_index=True)
    return df_all

# ============================== Sidebar ==============================

st.sidebar.header("⚙️ Options")

st.sidebar.subheader("Données")
mode_data = st.sidebar.radio(
    "Source des données",
    ["CSV combiné", "CSV séparés", "Édition (sans CSV)", "Démo"],
    horizontal=True
)

uploaded_combined = None
uploaded_acts = None
uploaded_miles = None
uploaded_inputs = None

if mode_data == "CSV combiné":
    uploaded_combined = st.sidebar.file_uploader("Importer le CSV combiné (id,title,start,end[,group,milestone,progress,depends_on])", type=["csv"])
    uploaded_inputs = st.sidebar.file_uploader("Importer inputs.csv (date,label) — optionnel", type=["csv"])
elif mode_data == "CSV séparés":
    uploaded_acts = st.sidebar.file_uploader("activities.csv (id,title,start,end[,group,progress,depends_on])", type=["csv"])
    uploaded_miles = st.sidebar.file_uploader("milestones.csv (id,title,date) ou (id,title,start,end)", type=["csv"])
    uploaded_inputs = st.sidebar.file_uploader("inputs.csv (date,label) — optionnel", type=["csv"])

st.sidebar.markdown("---")
show_dependencies = st.sidebar.checkbox("Afficher dépendances", value=True)
if show_dependencies:
    dep_conn_type = st.sidebar.selectbox("Type liaison dépendances", ["Courbe", "Orthogonale"], index=0)
    if dep_conn_type == "Courbe":
        dep_arrow_rad = st.sidebar.slider("Courbure flèches dépendances", -1.0, 1.0, 0.3, step=0.05)
        dep_connstyle = f"arc3,rad={dep_arrow_rad}"
    else:
        dep_connstyle = "angle3,angleA=0,angleB=90"
    dep_arrow_style = st.sidebar.selectbox("Forme flèches dépendances", ["-|>", "->", "-["], index=0)
    dep_arrow_color = st.sidebar.color_picker("Couleur flèches dépendances", "#000000")
    dep_arrow_alpha = st.sidebar.slider("Transparence flèches dépendances", 0.0, 1.0, 1.0, step=0.05)
    dep_arrow_lw = st.sidebar.slider("Épaisseur flèches dépendances", 0.5, 5.0, 1.0, step=0.5)
    dep_arrow_ms = st.sidebar.slider("Taille tête flèches dépendances", 5, 30, 10)
else:
    dep_connstyle = ""
    dep_arrow_style = "-|>"
    dep_arrow_color = "black"
    dep_arrow_alpha = 1.0
    dep_arrow_lw = 1.0
    dep_arrow_ms = 10
unit = st.sidebar.selectbox("Unité de temps", ["Jours", "Semaines"], index=1)
inclusive_duration = st.sidebar.checkbox("Durée inclusive (inclure le jour de fin)", value=True)
tabs = st.sidebar.tabs([
    "Axes & Grille",
    "Plage X",
    "Visuel",
    "Titres/Durées",
    "Dates & End-caps",
    "Jalons",
    "Inputs",
    "Axe X",
])

with tabs[0]:
    st.subheader("Axes & Grille")
    rot = st.slider("Rotation des dates (X)", 0, 90, 30, step=5)
    show_grid = st.checkbox("Quadrillage pointillé", value=False)
    grid_axis = st.selectbox("Grille sur", ["x", "both"], index=0)
    x_tick_step = st.number_input(
        "Pas des graduations majeures (jours/semaines)",
        min_value=1,
        max_value=30,
        value=1,
        step=1,
    )
    date_fmt = st.text_input("Format des dates (strftime)", "%d %b %Y")
    top_axis = st.selectbox(
        "Graduations secondaires (haut)",
        ["Aucune", "Mois", "Numéros de semaine"],
        index=0,
    )

with tabs[1]:
    st.subheader("Plage X")
    x_min_str = st.text_input("X min (AAAA-MM-JJ) — optionnel", "")
    x_max_str = st.text_input("X max (AAAA-MM-JJ) — optionnel", "")
    x_margin_ratio = st.slider("Marge aux extrémités (%)", 0, 20, 5, step=1) / 100.0

with tabs[2]:
    st.subheader("Visuel")
    row_bg = st.checkbox("Bandes horizontales de fond", value=False)
    row_bg_alpha = st.slider("Transparence bandes", 0.0, 0.5, 0.10, step=0.01)
    row_bg_height = st.slider("Épaisseur bande (0–1)", 0.1, 1.0, 0.6, step=0.05)
    bar_height = st.slider("Hauteur des barres", 0.1, 1.0, 0.4, step=0.05)
    legend_loc = st.selectbox(
        "Position de la légende",
        [
            "upper left",
            "upper right",
            "lower left",
            "lower right",
            "center left",
            "center right",
            "upper center",
            "lower center",
            "center",
        ],
        index=0,
    )

with tabs[3]:
    st.subheader("Titres / Durées")
    titles_above = st.checkbox("Titres au-dessus des barres", value=True)
    title_max_fs = st.slider("Taille max titres", 6, 18, 10)
    title_min_fs = st.slider("Taille min titres", 4, 12, 6)
    title_gap_px = st.slider("Décalage vertical titres (px)", 0, 20, 6)
    title_padding_px = st.slider("Marge interne titres (px)", 0, 20, 6)

    dur_in_bar = st.checkbox("Afficher la durée dans la barre", value=True)
    dur_font = st.slider("Taille police durée", 6, 14, 8)
    dur_fmt_days = st.text_input("Format durée (jours)", "{d} j")
    dur_fmt_weeks = st.text_input("Format durée (semaines)", "{w:.1f} sem")

with tabs[4]:
    st.subheader("Dates & End-caps")
    show_start_end = st.checkbox("Étiquettes date début/fin", value=True)
    date_label_offset = st.number_input("Décalage étiquettes (jours)", value=0.2, step=0.1, format="%.1f")
    draw_endcaps = st.checkbox("Traits verticaux aux extrémités", value=False)
    endcap_len = st.slider("Longueur end-caps (en Y)", 0.1, 1.0, 0.35, step=0.05)
    show_today_line = st.checkbox("Ligne aujourd’hui", value=True)

with tabs[5]:
    st.subheader("Jalons")
    milestones_vlines = st.checkbox("Lignes verticales jalons", value=False)
    ms_markersize = st.slider("Taille marqueurs jalons", 8, 36, 16, step=1)
    ms_offset = st.slider("Offset vertical jalons (axes fraction)", 0.0, 0.1, 0.04, step=0.005)
    anti_overlap = st.checkbox("Anti-chevauchement (titres/jalons/inputs)", value=False)
    ms_alt_extra = st.slider("Alternance offset jalons (ajout max)", 0.0, 0.08, 0.03, step=0.005)
    show_ms_dates = st.checkbox("Afficher date jalon", value=True)
    ms_date_pos = (
        st.selectbox("Position date jalon", ["Opposé à la flèche", "Près de la pointe"])
        if show_ms_dates
        else None
    )
    ms_date_offset = (
        st.slider("Offset vertical date jalon (axes fraction)", 0.0, 0.1, 0.02, step=0.005)
        if show_ms_dates
        else 0.02
    )
    ms_date_fmt = (
        st.text_input("Format date jalon", "%d/%m") if show_ms_dates else "%d/%m"
    )

with tabs[6]:
    st.subheader("Inputs")
    show_inputs = st.checkbox("Afficher inputs", value=True)
    inputs_position = st.selectbox(
        "Position des inputs",
        ["Haut (flèches descendantes)", "Bas (flèches montantes)"],
        index=0,
    )
    inputs_alt_extra = st.slider(
        "Alternance hauteur inputs (ajout max)", 0.0, 1.0, 0.4, step=0.05
    )
    inputs_top_margin = (
        st.slider("Marge verticale haut (Y)", 0.0, 2.0, 1.0, step=0.1)
        if inputs_position == "Haut (flèches descendantes)"
        else 1.0
    )
    inputs_bottom_margin = (
        st.slider("Marge verticale bas (Y)", 0.0, 2.0, 1.0, step=0.1)
        if inputs_position == "Bas (flèches montantes)"
        else 1.0
    )
    input_arrow_len = st.slider("Longueur flèche input (Y)", 0.2, 2.0, 0.8, step=0.1)
    show_input_dates = st.checkbox("Afficher date input", value=False)
    input_date_fmt = (
        st.text_input("Format date input", "%d/%m") if show_input_dates else "%d/%m"
    )
    input_date_offset = (
        st.slider("Décalage vertical date input (Y)", 0.0, 1.0, 0.15, step=0.05)
        if show_input_dates
        else 0.15
    )
    input_date_rot = (
        st.slider("Rotation date input", 0, 90, 90, step=5) if show_input_dates else 90
    )
    input_titles_in_legend = st.checkbox("Titres inputs en légende", value=False)
    input_title_date_split = st.checkbox("Date & titre de part et d'autre", value=False)
    input_lr_offset = (
        st.slider("Décalage horizontal date/titre (jours)", 0.0, 5.0, 0.5, step=0.1)
        if input_title_date_split
        else 0.5
    )

with tabs[7]:
    st.subheader("Axe X (flèche)")
    ax_arrow_lw = st.slider("Épaisseur flèche axe X", 2.0, 16.0, 8.0, step=0.5)
    ax_arrow_scale = st.slider("Taille pointe flèche", 10, 60, 30, step=2)

st.sidebar.markdown("---")
fig_w = st.sidebar.slider("Largeur figure", 6, 20, 12)
fig_h = st.sidebar.slider("Hauteur figure", 4, 12, 6)
st.sidebar.caption("💡 Télécharge le PNG en bas de page.")

# ============================== Meta Infos ==============================
st.sidebar.divider()
st.sidebar.caption("👤 Auteur : Ibrahim Bitar")
st.sidebar.caption("🏷️ Version : v1.1.0")
st.sidebar.caption("📅 Release : 04/09/2025")

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

        edit_csv = st.sidebar.checkbox("Éditer les données importées", value=False)
        if edit_csv:
            file_id = getattr(uploaded_combined, "name", "")
            if (st.session_state.get("editor_source") != "uploaded" or
                    st.session_state.get("editor_file") != file_id):
                _init_editor_state_from_df(df_all, df_inputs, file_id)

            st.subheader("✍️ Édition des données importées")
            st.markdown("**Activités** (id, title, start, end, group, progress, depends_on)")
            editor_tasks = st.data_editor(
                st.session_state.editor_tasks,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "id": st.column_config.TextColumn("id"),
                    "title": st.column_config.TextColumn("title", width="medium"),
                    "start": st.column_config.DateColumn("start"),
                    "end": st.column_config.DateColumn("end"),
                    "group": st.column_config.TextColumn("group", width="small"),
                    "progress": st.column_config.NumberColumn("progress", min_value=0, max_value=100, step=1),
                    "depends_on": st.column_config.TextColumn("depends_on", help="IDs séparés par des virgules"),
                },
                key="editor_tasks_uploaded",
            )

            st.markdown("**Jalons** (id, title, date, marker)")
            editor_ms = st.data_editor(
                st.session_state.editor_ms,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "id": st.column_config.TextColumn("id"),
                    "title": st.column_config.TextColumn("title", width="medium"),
                    "date": st.column_config.DateColumn("date"),
                    "marker": st.column_config.TextColumn(
                        "marker", help="Symbole Matplotlib (v, o, s, ^, ...)", width="small"
                    ),
                },
                key="editor_ms_uploaded",
            )

            with st.expander("Inputs (optionnel) — date, label", expanded=False):
                editor_inputs = st.data_editor(
                    st.session_state.editor_inputs,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "date": st.column_config.DateColumn("date"),
                        "label": st.column_config.TextColumn("label", width="medium"),
                    },
                    key="editor_inputs_uploaded",
                )

            c1, c2 = st.columns([1, 1])
            with c1:
                apply_btn = st.button("✅ Appliquer", type="primary", key="apply_upload_edit")
            with c2:
                export_btn = st.button("⬇️ Exporter CSV", key="export_upload_edit")

            if apply_btn:
                errs = _validate_editors(editor_tasks.copy(), editor_ms.copy())
                if errs:
                    for e in errs:
                        st.error(e)
                    st.stop()
                st.session_state.editor_tasks = editor_tasks
                st.session_state.editor_ms = editor_ms
                st.session_state.editor_inputs = editor_inputs
                st.success("Modifications appliquées.")

            if export_btn:
                df_export = _build_df_all_from_editors(editor_tasks, editor_ms)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_export.to_excel(writer, sheet_name="planning", index=False)
                    editor_inputs.to_excel(writer, sheet_name="inputs", index=False)
                buffer.seek(0)
                st.download_button(
                    "⬇️ Télécharger donnees.xlsx",
                    buffer.getvalue(),
                    "donnees.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            df_all = _build_df_all_from_editors(st.session_state.editor_tasks, st.session_state.editor_ms)
            df_inputs = (
                st.session_state.editor_inputs
                if not st.session_state.editor_inputs.empty
                else pd.DataFrame(columns=["date", "label"])
            )
        
    elif mode_data == "Édition (sans CSV)":
        _init_editor_state()
        st.subheader("✍️ Édition des données (sans CSV)")

        st.markdown("**Activités** (id, title, start, end, group, progress, depends_on)")
        editor_tasks = st.data_editor(
            st.session_state.editor_tasks,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("id"),
                "title": st.column_config.TextColumn("title", width="medium"),
                "start": st.column_config.DateColumn("start"),
                "end": st.column_config.DateColumn("end"),
                "group": st.column_config.TextColumn("group", width="small"),
                "progress": st.column_config.NumberColumn("progress", min_value=0, max_value=100, step=1),
                "depends_on": st.column_config.TextColumn("depends_on", help="IDs séparés par des virgules")
            },
            key="editor_tasks_widget"
        )

        st.markdown("**Jalons** (id, title, date, marker)")
        editor_ms = st.data_editor(
            st.session_state.editor_ms,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("id"),
                "title": st.column_config.TextColumn("title", width="medium"),
                "date": st.column_config.DateColumn("date"),
                "marker": st.column_config.TextColumn("marker", help="Symbole Matplotlib (v, o, s, ^, ...)", width="small"),
            },
            key="editor_ms_widget",
        )

        with st.expander("Inputs (optionnel) — date, label", expanded=False):
            editor_inputs = st.data_editor(
                st.session_state.editor_inputs,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "date": st.column_config.DateColumn("date"),
                    "label": st.column_config.TextColumn("label", width="medium")
                },
                key="editor_inputs_widget"
            )

        # Actions
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            apply_btn = st.button("✅ Appliquer", type="primary")
        with c2:
            export_btn = st.button("⬇️ Exporter CSV")

        # Validation & application
        if apply_btn:
            errs = _validate_editors(editor_tasks.copy(), editor_ms.copy())
            if errs:
                for e in errs:
                    st.error(e)
                st.stop()
            st.session_state.editor_tasks = editor_tasks
            st.session_state.editor_ms = editor_ms
            st.session_state.editor_inputs = editor_inputs
            st.success("Modifications appliquées.")

        # Exports (CSV)
        if export_btn:
            tasks_csv = editor_tasks.to_csv(index=False).encode("utf-8")
            ms_csv = editor_ms.to_csv(index=False).encode("utf-8")
            inputs_csv = editor_inputs.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger activities.csv", tasks_csv, "activities.csv", "text/csv")
            st.download_button("⬇️ Télécharger milestones.csv", ms_csv, "milestones.csv", "text/csv")
            st.download_button("⬇️ Télécharger inputs.csv", inputs_csv, "inputs.csv", "text/csv")

        # Construire df_all / df_inputs à partir des éditeurs
        df_all = _build_df_all_from_editors(st.session_state.editor_tasks, st.session_state.editor_ms)
        df_inputs = st.session_state.editor_inputs if not st.session_state.editor_inputs.empty else pd.DataFrame(columns=["date","label"])
            
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
        if "progress" not in df_act: df_act["progress"] = 0
        df_act["progress"] = pd.to_numeric(df_act["progress"], errors="coerce").fillna(0)
        df_act["milestone"] = 0
        df_act["marker"] = df_act.get("marker", "")
        df_act["marker"] = df_act["marker"].fillna("").astype(str)
        if "marker" not in df_ms:
            df_ms["marker"] = "v"
        df_ms["progress"] = 0
        df_all = pd.concat([
            df_act[["id","title","start","end","group","milestone","progress","depends_on","marker"]],
            df_ms[["id","title","start","end","marker","milestone"]]
                .assign(group="", progress=0, depends_on="")[
                    ["id","title","start","end","group","milestone","progress","depends_on","marker"]
                ],
        ], ignore_index=True)
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

# Parse dates and progress
df_all["start"] = pd.to_datetime(df_all["start"]).dt.date
df_all["end"]   = pd.to_datetime(df_all["end"]).dt.date
df_all["progress"] = pd.to_numeric(df_all.get("progress", 0), errors="coerce").fillna(0)
df_all = df_all.sort_values("start").reset_index(drop=True)
df_ms = df_all[(df_all.get("milestone", 0).astype(int) == 1) | (df_all["start"] == df_all["end"])].copy()
df_tasks = df_all[~df_all.index.isin(df_ms.index)].copy()

df_ms["marker"] = df_ms.get("marker", "v")
df_ms["marker"] = df_ms["marker"].replace("", "v").fillna("v").astype(str)
df_tasks["marker"] = df_tasks.get("marker", "")
df_tasks["marker"] = df_tasks["marker"].fillna("").astype(str)

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

# Palette et couleurs de groupes
group_palette = plt.get_cmap("tab20").colors
unique_groups = df_tasks["group"].dropna().unique()
group_colors = {g: group_palette[i % len(group_palette)] for i, g in enumerate(unique_groups)}
default_color = "gray"
legend_handles = []

# Bars + labels
task_positions = {}
for i, row in enumerate(df_tasks.itertuples(index=False), start=1):
    start_num = mdates.date2num(row.start)
    end_num   = mdates.date2num(row.end)
    duration_num = end_num - start_num
    duration_days = days_between(row.start, row.end, inclusive=inclusive_duration)
    duration_weeks = duration_days / 7.0
    task_positions[row.id] = (start_num, end_num, i)

    progress = getattr(row, "progress", 0)
    try:
        progress = float(progress)
    except Exception:
        progress = 0.0
    progress = max(0.0, min(progress / 100.0, 1.0))

    base_color = group_colors.get(row.group, default_color)
    # background bar tinted by group color
    ax.barh(
        i,
        duration_num,
        left=start_num,
        height=bar_height,
        align='center',
        color=base_color,
        alpha=0.3,
        zorder=2,
    )
    # progress bar (full color showing advancement)
    ax.barh(
        i,
        duration_num * progress,
        left=start_num,
        height=bar_height,
        align='center',
        color=base_color,
        zorder=3,
    )

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

# Dependencies arrows
if show_dependencies:
    for row in df_tasks.itertuples(index=False):
        deps = [d.strip() for d in getattr(row, "depends_on", "").split(",") if d.strip()]
        if not deps:
            continue
        cur = task_positions.get(row.id)
        if not cur:
            continue
        start_cur, end_cur, y_cur = cur
        for dep_id in deps:
            prev = task_positions.get(dep_id)
            if not prev:
                continue
            _, end_prev, y_prev = prev
            ax.annotate(
                '',
                xy=(start_cur, y_cur),
                xytext=(end_prev, y_prev),
                arrowprops=dict(
                    arrowstyle=dep_arrow_style,
                    color=dep_arrow_color,
                    alpha=dep_arrow_alpha,
                    lw=dep_arrow_lw,
                    connectionstyle=dep_connstyle,
                    mutation_scale=dep_arrow_ms,
                ),
                zorder=3,
            )

# Légende des groupes
for g, c in group_colors.items():
    legend_handles.append(Line2D([0], [0], color=c, linewidth=6, label=g))

# Milestones at y=0
palette = plt.get_cmap("tab10").colors
blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)

for k, row in enumerate(df_ms.itertuples(index=False)):
    x_ms = mdates.date2num(row.start)
    # vline
    if milestones_vlines:
        ax.axvline(x=x_ms, linestyle=":", alpha=0.3, zorder=3)
    # alternating vertical offset (axes fraction)
    extra = (ms_alt_extra if (anti_overlap and k % 2 == 1) else 0.0)
    y_af = ms_offset + extra
    color = palette[k % len(palette)]
    marker = getattr(row, "marker", "v")
    ax.plot(
        [x_ms],
        [y_af],
        marker=marker,
        markersize=ms_markersize,
        markerfacecolor=color,
        markeredgecolor=color,
        transform=blended,
        zorder=7,
    )
    if show_ms_dates:
        txt = mdates.num2date(x_ms).strftime(ms_date_fmt)
        gap = ms_date_offset
        if ms_date_pos == "Opposé à la flèche":
            y_txt = y_af + gap
            va = "bottom"
        else:
            y_txt = y_af - gap
            va = "top"

        ax.text(
            x_ms,
            y_txt,
            txt,
            rotation=90,
            va=va,
            ha="center",
            fontsize=9,
            transform=blended,
            zorder=7,
            clip_on=False,
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markersize=ms_markersize,
            markerfacecolor=color,
            markeredgecolor=color,
            label=row.title,
        )
    )

# Inputs (arrows for inputs)
if show_inputs and not df_inputs.empty:
    for k, row in enumerate(df_inputs.itertuples(index=False)):
        x_d = mdates.date2num(row.date)
        extra = inputs_alt_extra if (anti_overlap and k % 2 == 1) else 0.0
        if inputs_position == "Haut (flèches descendantes)":
            y_base = len(df_tasks) + inputs_top_margin
            y0 = y_base + extra
            y1 = y0 - input_arrow_len
            va_lab = "top"
            date_sign = -1.0
        else:
            y_base = -inputs_bottom_margin
            y0 = y_base - extra
            y1 = y0 + input_arrow_len
            va_lab = "bottom"
            date_sign = 1.0

        ax.annotate(
            "",
            xy=(x_d, y1),
            xytext=(x_d, y0),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(arrowstyle='-|>', lw=1.5),
            zorder=6,
        )
        x_label = x_d - (input_lr_offset if (input_title_date_split and not input_titles_in_legend) else 0.0)
        x_date = x_d + (input_lr_offset if input_title_date_split else 0.0)

        if input_titles_in_legend:
            marker = "v" if inputs_position == "Haut (flèches descendantes)" else "^"
            legend_handles.append(Line2D([0],[0], marker=marker, linestyle="None",
                                         markersize=ms_markersize,
                                         markerfacecolor="black", markeredgecolor="black",
                                         label=row.label))
        else:
            ax.text(
                x_label,
                y1 + date_sign * 0.05,
                row.label,
                rotation=90,
                va=va_lab,
                ha="center",
                fontsize=9,
                zorder=6,
                clip_on=False,
            )

        if show_input_dates:
            txt = mdates.num2date(x_d).strftime(input_date_fmt)
            ax.text(
                x_date,
                y1 + date_sign * input_date_offset,
                txt,
                rotation=input_date_rot,
                va=va_lab,
                ha="center",
                fontsize=9,
                zorder=6,
                clip_on=False,
            )

if legend_handles:
    ax.legend(handles=legend_handles, loc=legend_loc, frameon=False)

# X axis limits and ticks
ax.set_xlim(x_min_num, x_max_num)
if show_today_line:
    today_num = mdates.date2num(pd.Timestamp.today().date())
    ax.axvline(today_num, color='red', linestyle='--', alpha=0.5, zorder=1)
y_min, y_max = -0.5, max(2.0, len(df_tasks) + inputs_top_margin + 0.8)
if show_inputs and not df_inputs.empty:
    if inputs_position == "Haut (flèches descendantes)":
        y_needed = len(df_tasks) + inputs_top_margin + input_arrow_len + inputs_alt_extra
        if show_input_dates:
            y_needed += input_date_offset
        y_max = max(y_max, y_needed + 0.5)
    else:
        y_needed = -inputs_bottom_margin - input_arrow_len - inputs_alt_extra
        if show_input_dates:
            y_needed -= input_date_offset
        y_min = min(y_min, y_needed - 0.5)
ax.set_ylim(y_min, y_max)
ax.xaxis_date()
if unit == "Jours":
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=int(x_tick_step)))
else:
    # step in weeks
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=int(x_tick_step)))
ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
if top_axis != "Aucune":
    sec_ax = ax.secondary_xaxis('top')
    sec_ax.set_xlim(x_min_num, x_max_num)
    if top_axis == "Mois":
        sec_ax.xaxis.set_major_locator(mdates.MonthLocator())
        sec_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        sec_ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        sec_ax.xaxis.set_major_formatter(mdates.DateFormatter("%W"))
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

