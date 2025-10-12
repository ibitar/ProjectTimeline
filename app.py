# app.py
# Streamlit Gantt Planner — Ibrahim's options
# Run: streamlit run app.py

import io
import math
import unicodedata
from html import escape
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from datetime import datetime, date, timedelta, timezone
from itertools import zip_longest

st.set_page_config(page_title="Gantt Planner", layout="wide")


def _normalize_column_name(name: str) -> str:
    """Return a lowercase, accent-free version of a column/sheet name."""

    if not isinstance(name, str):
        name = "" if name is None else str(name)
    normalized = unicodedata.normalize("NFD", name)
    stripped = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return stripped.lower().strip()


def _find_column(cols: dict, *candidates: str) -> str | None:
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in cols:
            return cols[key]
    return None

# ============================== Helpers ==============================

def days_between(d0, d1, inclusive=True):
    """Return duration in days between two python dates."""
    delta = (d1 - d0).days
    return delta + 1 if inclusive else delta

def weeks_between(d0, d1, inclusive=True):
    d = days_between(d0, d1, inclusive=inclusive)
    return d / 7.0

def _ensure_busdaycalendar(busdaycal):
    """Return a numpy.busdaycalendar from a variety of inputs or None."""

    if busdaycal is None:
        return None

    if isinstance(busdaycal, np.busdaycalendar):
        return busdaycal

    # Accept raw holiday collections (NumPy arrays, Pandas objects, Python iterables).
    holidays_source = None

    if isinstance(busdaycal, dict):
        holidays_source = busdaycal.get("holidays")
    else:
        holidays_source = busdaycal

    try:
        holidays = pd.to_datetime(holidays_source, errors="coerce")
    except Exception:
        return None

    if hasattr(holidays, "dropna"):
        holidays = holidays.dropna()
    else:
        holidays = pd.Series(holidays).dropna()

    if len(holidays) == 0:
        return None

    try:
        holidays_np = holidays.to_numpy(dtype="datetime64[D]")
    except AttributeError:
        holidays_np = np.array(holidays, dtype="datetime64[D]")

    if holidays_np.size == 0:
        return None

    try:
        return np.busdaycalendar(holidays=holidays_np)
    except Exception:
        return None


def business_days_between(d0, d1, inclusive=True, busdaycal=None):
    """Return number of working days between two dates, excluding weekends/holidays."""
    start = to_date(d0)
    end = to_date(d1)
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return 0

    d0_np = np.datetime64(start, "D")
    d1_np = np.datetime64(end, "D")
    if d1_np < d0_np:
        d0_np, d1_np = d1_np, d0_np

    normalized_calendar = _ensure_busdaycalendar(busdaycal)

    # n'appelle busday_count/is_busday avec busdaycal QUE s'il n'est pas None
    if normalized_calendar is None:
        bd = np.busday_count(d0_np, d1_np)
        if inclusive and np.is_busday(d1_np):
            bd += 1
    else:
        bd = np.busday_count(d0_np, d1_np, busdaycal=normalized_calendar)
        if inclusive and np.is_busday(d1_np, busdaycal=normalized_calendar):
            bd += 1

    return int(bd)

def leave_zones_to_holidays(zones):
    """Convert configured leave zones to business-day holidays for NumPy calendars."""
    if not zones:
        return None

    holidays = []
    for zone in zones:
        start = to_date(zone.get("start")) if isinstance(zone, dict) else None
        end = to_date(zone.get("end")) if isinstance(zone, dict) else None
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        rng = pd.bdate_range(start=start, end=end)
        if not rng.empty:
            holidays.append(rng.to_numpy(dtype="datetime64[D]"))

    if not holidays:
        return None

    all_holidays = np.unique(np.concatenate(holidays))
    return all_holidays

def to_date(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if pd.isna(x):
        return None

    if isinstance(x, pd.Timestamp):
        try:
            if x.tzinfo is not None:
                x = x.tz_convert("UTC").tz_localize(None)
        except AttributeError:
            pass
        return x.date()

    if isinstance(x, datetime):
        if x.tzinfo is not None:
            x = x.astimezone(timezone.utc).replace(tzinfo=None)
        return x.date()

    if isinstance(x, date):
        return x

    try:
        parsed = pd.to_datetime(x, errors="coerce")
    except Exception:
        try:
            parsed = pd.to_datetime(str(x), errors="coerce")
        except Exception:
            return None

    if pd.isna(parsed):
        return None

    if isinstance(parsed, datetime):
        return parsed.date()
    if isinstance(parsed, date):
        return parsed

    try:
        return pd.to_datetime(parsed, errors="coerce").date()
    except Exception:
        return None


def _normalize_planning_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = {_normalize_column_name(c): c for c in df.columns}
    required = {"id", "title", "start", "end"}
    missing = [r for r in required if r not in cols]
    if missing:
        raise ValueError(
            "Fichier combiné attendu avec colonnes au minimum: id,title,start,end[,group,milestone,progress,depends_on]"
        )

    rename_map = {}
    for key in ["id", "title", "start", "end", "group", "milestone", "progress", "depends_on", "marker"]:
        src = cols.get(key)
        if src:
            rename_map[src] = key

    df_norm = df.rename(columns=rename_map).copy()
    if "group" not in df_norm:
        df_norm["group"] = ""
    if "milestone" not in df_norm:
        df_norm["milestone"] = 0
    if "progress" not in df_norm:
        df_norm["progress"] = 0
    if "depends_on" not in df_norm:
        df_norm["depends_on"] = ""
    if "marker" not in df_norm:
        df_norm["marker"] = ""

    df_norm["id"] = df_norm["id"].astype(str)
    df_norm["title"] = df_norm["title"].astype(str)
    df_norm["group"] = df_norm["group"].fillna("").astype(str)
    df_norm["depends_on"] = df_norm["depends_on"].fillna("").astype(str)
    df_norm["marker"] = df_norm["marker"].fillna("").astype(str)
    df_norm["progress"] = pd.to_numeric(df_norm["progress"], errors="coerce").fillna(0)

    return df_norm


def _normalize_inputs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "label"])

    cols = {_normalize_column_name(c): c for c in df.columns}
    date_col = _find_column(cols, "date")
    label_col = _find_column(cols, "label", "libelle", "titre")
    if not date_col or not label_col:
        raise ValueError("Le fichier inputs doit contenir: date,label")

    df_norm = df.rename(columns={date_col: "date", label_col: "label"})
    return df_norm[["date", "label"]]


def _normalize_leave_zones_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["start", "end", "label"])

    cols = {_normalize_column_name(c): c for c in df.columns}
    start_col = _find_column(cols, "start", "debut", "début", "date_debut")
    end_col = _find_column(cols, "end", "fin", "date_fin")
    label_col = _find_column(cols, "label", "titre", "libelle", "libellé")
    if not start_col or not end_col:
        raise ValueError("Le fichier congés doit contenir: start,end[,label]")

    rename_map = {start_col: "start", end_col: "end"}
    if label_col:
        rename_map[label_col] = "label"
    df_norm = df.rename(columns=rename_map)

    for col in ("start", "end"):
        df_norm[col] = pd.to_datetime(df_norm[col], errors="coerce")

    df_norm = df_norm.dropna(subset=["start", "end"])
    if df_norm.empty:
        return pd.DataFrame(columns=["start", "end", "label"])

    df_norm["start"] = df_norm["start"].dt.date
    df_norm["end"] = df_norm["end"].dt.date

    if "label" not in df_norm:
        df_norm["label"] = "Congés"
    else:
        df_norm["label"] = df_norm["label"].apply(
            lambda v: "Congés" if (pd.isna(v) or not str(v).strip()) else str(v)
        )

    return df_norm[["start", "end", "label"]]


def _normalize_holidays_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])

    cols = {_normalize_column_name(c): c for c in df.columns}
    date_col = _find_column(cols, "date")
    if not date_col:
        raise ValueError("Le fichier jours_feries doit contenir: date")

    df_norm = df.rename(columns={date_col: "date"})
    return df_norm[["date"]]


def _normalize_leave_zones_state():
    """Ensure the leave zones stored in the session are a clean DataFrame."""

    today = pd.Timestamp(date.today())
    default_row = {"start": today, "end": today, "label": "Congés"}

    raw_value = st.session_state.get("leave_zones")

    if raw_value is None:
        df = pd.DataFrame([default_row])
    elif isinstance(raw_value, pd.DataFrame):
        df = raw_value.copy()
    elif isinstance(raw_value, dict):
        df = pd.DataFrame([raw_value])
    elif isinstance(raw_value, (list, tuple)):
        try:
            df = pd.DataFrame(raw_value)
        except (TypeError, ValueError):
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        df = pd.DataFrame([default_row])

    for col in ("start", "end"):
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce").fillna(today)
        else:
            df[col] = today

    if "label" in df:
        def _clean_label(value):
            if isinstance(value, str):
                return value.strip() or "Congés"
            if pd.isna(value):
                return "Congés"
            return str(value)

        df["label"] = df["label"].apply(_clean_label)
    else:
        df["label"] = "Congés"

    st.session_state.leave_zones = df[["start", "end", "label"]]


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

def _read_table(file) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file."""
    name = getattr(file, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def parse_combined_file(file) -> tuple[pd.DataFrame, dict]:
    name = getattr(file, "name", "").lower()
    extras: dict[str, pd.DataFrame] = {}

    if name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(file, sheet_name=None)
        planning_df = None
        for sheet_name, sheet_df in sheets.items():
            try:
                planning_df = _normalize_planning_dataframe(sheet_df)
                extras["planning_sheet"] = sheet_name
                break
            except ValueError:
                continue
        if planning_df is None:
            raise ValueError(
                "Impossible de trouver une feuille avec les colonnes id,title,start,end dans le fichier combiné."
            )

        # Optional sheets
        sheet_lookup = {_normalize_column_name(name): (name, df) for name, df in sheets.items()}
        for _, aliases, normalizer, store_key in [
            ("inputs", ("inputs", "annotations"), _normalize_inputs_dataframe, "inputs"),
            ("leave_zones", ("leave_zones", "conges", "congés", "vacances"), _normalize_leave_zones_dataframe, "leave_zones"),
            ("holidays", ("jours_feries", "joursferies", "holidays"), _normalize_holidays_dataframe, "holidays"),
        ]:
            for alias in aliases:
                alias_key = _normalize_column_name(alias)
                if alias_key in sheet_lookup:
                    raw_df = sheet_lookup[alias_key][1]
                    try:
                        extras[store_key] = normalizer(raw_df)
                    except ValueError:
                        pass
                    break

        return planning_df, extras

    df = _read_table(file)
    planning_df = _normalize_planning_dataframe(df)
    return planning_df, extras

def parse_activities_file(file) -> pd.DataFrame:
    df = _read_table(file)
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})
    needed = {"id","title","start","end"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(
            "Le fichier activities doit contenir: id,title,start,end[,group,progress,depends_on]"
        )
    if "progress" not in df:
        df["progress"] = 0
    df["progress"] = pd.to_numeric(df["progress"], errors="coerce").fillna(0)
    return df

def parse_milestones_file(file) -> pd.DataFrame:
    df = _read_table(file)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # accept either (id,title,date) or (id,title,start,end with same day)
    if {"id", "title", "date"}.issubset(df.columns):
        df["start"] = df["date"]
        df["end"] = df["date"]
    elif {"id", "title", "start", "end"}.issubset(df.columns):
        pass
    else:
        raise ValueError(
            "Le fichier milestones doit contenir: id,title,date OU id,title,start,end (start=end)."
        )
    if "marker" not in df:
        df["marker"] = "v"
    df["marker"] = df["marker"].replace("", "v").fillna("v").astype(str)
    df["milestone"] = 1
    return df

def parse_inputs_file(file) -> pd.DataFrame:
    df = _read_table(file)
    return _normalize_inputs_dataframe(df)

def parse_holidays_file(file) -> pd.DataFrame:
    df = _read_table(file)
    return _normalize_holidays_dataframe(df)

import datetime as dt

def _init_editor_state():
    if "editor_tasks" not in st.session_state:
        st.session_state.editor_tasks = pd.DataFrame([
            {"id": "T1", "title": "Cadrage",
             "start": dt.date(2025, 9, 1),
             "end": dt.date(2025, 9, 5),
             "group": "Pilotage", "progress": 0, "depends_on": ""},
            {"id": "T2", "title": "Spécifications",
             "start": dt.date(2025, 9, 8),
             "end": dt.date(2025, 9, 19),
             "group": "Conception", "progress": 0, "depends_on": "T1"},
        ], columns=["id", "title", "start", "end", "group", "progress", "depends_on"])

    if "editor_ms" not in st.session_state:
        st.session_state.editor_ms = pd.DataFrame([
            {
                "id": "M1",
                "title": "Specs validées",
                "date": dt.date(2025, 9, 22),
                "marker": "v",
            },
        ], columns=["id", "title", "date", "marker"])

    if "editor_inputs" not in st.session_state:
        st.session_state.editor_inputs = pd.DataFrame([
            {"date": dt.date(2025, 9, 10), "label": "Input A"},
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
# Titre de l'application dans la barre latérale
st.sidebar.title("Gantt Planner")
search_term = st.sidebar.text_input("Rechercher une option", help="Filtre les options disponibles")
st.sidebar.header("⚙️ Options")

import unicodedata

def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn").lower()

def expander_search(label: str, keywords=None, default=False, **kwargs):
    keys = [label] + (keywords or [])
    expanded = default or (bool(search_term) and any(_norm(search_term) in _norm(k) for k in keys))
    return st.expander(label, expanded=expanded, **kwargs)

SEARCH_CATALOG = {
    "Grille": "Axe X",
    "Plage temporelle": "Axe X",
    "Flèche": "Axe X",
    "Barres": "Activités",
    "Titres & Durées": "Activités",
    "Dates & end-caps": "Activités",
    "Inputs (optionnel) — date, label": "Annotations",
    "Zone hachurée": "Annotations",
    "Couleurs des groupes": "Activités",
}

if search_term:
    matches = {k: v for k, v in SEARCH_CATALOG.items() if _norm(search_term) in _norm(k)}
    if matches:
        st.sidebar.markdown("**Options trouvées :**")
        for k, v in matches.items():
            st.sidebar.markdown(f"- {k} *(dans {v})*")
    else:
        st.sidebar.markdown("Aucune option correspondante.")

st.sidebar.subheader("Données")
mode_data = st.sidebar.radio(
    "Source des données",
    ["Fichier combiné", "Fichiers séparés", "Édition (sans fichier)", "Démo"],
    horizontal=True,
    help="Choisir la provenance des données"
)

uploaded_combined = None
uploaded_acts = None
uploaded_miles = None
uploaded_inputs = None
uploaded_holidays = None

if mode_data == "Fichier combiné":
    uploaded_combined = st.sidebar.file_uploader(
        "Importer le fichier combiné (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        help="Charger un fichier unique contenant toutes les données",
    )
    uploaded_inputs = st.sidebar.file_uploader(
        "Importer inputs (CSV/Excel) — optionnel",
        type=["csv", "xlsx", "xls"],
        help="Ajouter des informations d'inputs",
    )
elif mode_data == "Fichiers séparés":
    uploaded_acts = st.sidebar.file_uploader(
        "activities (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        help="Importer les activités du projet",
    )
    uploaded_miles = st.sidebar.file_uploader(
        "milestones (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        help="Importer les jalons du projet",
    )
    uploaded_inputs = st.sidebar.file_uploader(
        "inputs (CSV/Excel) — optionnel",
        type=["csv", "xlsx", "xls"],
        help="Ajouter des inputs au planning",
    )

uploaded_holidays = st.sidebar.file_uploader(
    "jours_feries (CSV/Excel) — optionnel",
    type=["csv", "xlsx", "xls"],
    help="Spécifier les jours fériés",
)

st.sidebar.markdown("---")
planning_title = st.sidebar.text_input("Titre du planning", "Planning")
planning_title_size = st.sidebar.slider("Taille du titre", 8, 32, 32)
planning_title_color = st.sidebar.color_picker("Couleur du titre", "#000000")
planning_title_show_today = st.sidebar.checkbox(
    "Afficher la date du jour sous le titre", value=False
)
planning_title_date_format = st.sidebar.text_input(
    "Format de la date du titre", "%d/%m/%Y"
)
default_subtitle_size = max(6, min(planning_title_size - 8, planning_title_size))
planning_title_subtitle_size = st.sidebar.slider(
    "Taille du sous-titre (date)",
    6,
    28,
    value=default_subtitle_size,
)
unit = st.sidebar.selectbox(
    "Unité de temps",
    ["Jours", "Semaines"],
    index=0,
    help="Choisir l'unité d'affichage des durées",
)
inclusive_duration = st.sidebar.checkbox(
    "Durée inclusive (inclure le jour de fin)",
    value=True,
    help="Inclure le jour de fin dans la durée",
)
show_assumptions_note = st.sidebar.checkbox(
    "Afficher une note d'hypothèses",
    value=False,
    help="Ajoute sous le planning un rappel des hypothèses retenues (ex. calcul en jours ouvrés).",
)
assumptions_note_key = "planning_assumptions_note"
assumptions_note_default = (
    "- Les durées affichées excluent les week-ends, les jours fériés fournis et les périodes de congés configurées."
)
if assumptions_note_key not in st.session_state:
    st.session_state[assumptions_note_key] = assumptions_note_default

if show_assumptions_note:
    st.sidebar.text_area(
        "Hypothèses du planning",
        key=assumptions_note_key,
        height=120,
        help="Rédige les hypothèses à afficher sous le diagramme.",
    )

assumptions_note_text = st.session_state.get(assumptions_note_key, assumptions_note_default)

tabs = st.sidebar.tabs([
    "Axe X",
    "Activités",
    "Annotations",
    "Dépendances",
])

with tabs[0]:
    st.subheader("Axe X")
    with expander_search("Grille", ["Rotation des dates", "Quadrillage", "week-ends", "Grille sur", "graduations", "Format des dates"]):
        rot = st.slider("Rotation des dates (X)", 0, 90, 30, step=5, help="Ajuste l'angle des dates sur l'axe X")
        show_grid = st.checkbox("Quadrillage pointillé", value=True, help="Affiche un quadrillage en pointillés")
        highlight_weekends = st.checkbox("Surligner week-ends", value=False, help="Met en évidence les week-ends")
        grid_axis = st.selectbox("Grille sur", ["x", "both"], index=0, help="Choix de l'axe pour la grille")
        x_tick_step = st.number_input(
            "Pas des graduations majeures (jours/semaines)",
            min_value=1,
            max_value=30,
            value=2,
            step=1,
            help="Intervalle entre les graduations principales",
        )
        _date_fmt_options = {
            "31 Dec 2025": "%d %b %Y",
            "31/12/2025": "%d/%m/%Y",
            "Dec 31, 2025": "%b %d, %Y",
        }
        _date_fmt_label = st.selectbox(
            "Format des dates",
            list(_date_fmt_options.keys()) + ["Personnalisé…"],
            index=0,
            help="Format d'affichage des dates",
        )
        date_fmt = (
            st.text_input(
                "Format des dates (strftime)",
                "%d %b %Y",
                help="Format d'affichage des dates",
            )
            if _date_fmt_label == "Personnalisé…"
            else _date_fmt_options[_date_fmt_label]
        )
        top_axis = st.selectbox(
            "Graduations secondaires (haut)",
            ["Aucune", "Mois", "Numéros de semaine"],
            index=1,
            help="Ajoute une échelle secondaire en haut",
        )
    with expander_search("Plage temporelle", ["X min", "X max", "Marge aux extrémités"]):
        x_min_val = st.date_input(
            "X min — optionnel",
            value=None,
            format="YYYY-MM-DD",
            help="Date minimale affichée sur l'axe X",
        )
        x_max_val = st.date_input(
            "X max — optionnel",
            value=None,
            format="YYYY-MM-DD",
            help="Date maximale affichée sur l'axe X",
        )
        x_margin_ratio = (
            st.slider("Marge aux extrémités (%)", 0, 20, 5, step=1, help="Marge ajoutée aux extrémités") / 100.0
        )
    with expander_search("Flèche", ["Couleur flèche", "Épaisseur flèche", "Taille pointe"]):
        show_ax_arrow = st.checkbox("Afficher la flèche", value=True, help="Affiche une flèche indiquant le sens du temps")
        ax_arrow_color = st.color_picker("Couleur flèche", "#FF0000", help="Couleur de la flèche de l'axe X")
        ax_arrow_lw = st.slider("Épaisseur flèche axe X", 2.0, 16.0, 8.0, step=0.5, help="Épaisseur de la flèche")
        ax_arrow_scale = st.slider("Taille pointe flèche", 10, 60, 30, step=2, help="Taille de la pointe de la flèche")

with tabs[1]:
    st.subheader("Activités")
    with expander_search("Barres", ["Bandes", "Hauteur", "légende"]):
        row_bg = st.checkbox("Bandes horizontales de fond", value=False, help="Ajoute des bandes colorées derrière les activités")
        row_bg_alpha = st.slider("Transparence bandes", 0.0, 0.5, 0.10, step=0.01, help="Définit la transparence des bandes")
        row_bg_height = st.slider("Épaisseur bande (0–1)", 0.1, 1.0, 0.6, step=0.05, help="Hauteur des bandes de fond")
        bar_height = st.slider("Hauteur des barres", 0.1, 1.0, 0.4, step=0.05, help="Épaisseur des barres d'activités")
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
                "outside right",
                "outside left",
                "outside bottom",
                "outside top",
            ],
            index=9,
            help="Emplacement de la légende du graphique",
        )
    with expander_search("Titres & Durées", ["Titre", "Durée", "progression"]):
        titles_above = st.checkbox("Titres au-dessus des barres", value=True, help="Place les titres au-dessus des barres")
        title_max_fs = st.slider("Taille max titres", 6, 18, 10, help="Taille maximale des titres")
        title_min_fs = st.slider("Taille min titres", 4, 12, 6, help="Taille minimale des titres")
        title_gap_px = st.slider("Décalage vertical titres (px)", 0, 20, 6, help="Décalage vertical des titres")
        title_padding_px = st.slider("Marge interne titres (px)", 0, 20, 6, help="Marge autour du texte du titre")
        title_align_start = st.checkbox(
            "Aligner les titres sur le début",
            value=False,
            help="Si coché, les titres commencent au début de la barre au lieu d'être centrés",
        )

        dur_in_bar = st.checkbox(
            "Afficher la durée dans la barre",
            value=True,
            help="Affiche la durée à l'intérieur de la barre",
        )
        dur_font = st.slider("Taille police durée", 6, 14, 8, help="Taille de police pour la durée")
        dur_unit = st.radio(
            "Unité de la durée",
            ["Jours", "Semaines"],
            index=0 if unit == "Jours" else 1,
            help="Choisir l'unité d'affichage de la durée",
        )
        if dur_unit == "Jours":
            dur_fmt = st.text_input(
                "Format durée",
                "{d} j",
                help="Format pour la durée en jours",
            )
        else:
            dur_fmt = st.text_input(
                "Format durée",
                "{w:.1f} sem",
                help="Format pour la durée en semaines",
            )
    with expander_search("Dates & end-caps", ["dates", "end-caps"]):
        show_start_end = st.checkbox("Étiquettes date début/fin", value=True, help="Affiche les dates de début et fin des tâches")
        date_label_offset = st.number_input(
            "Décalage étiquettes (jours)",
            value=0.2,
            step=0.1,
            format="%.1f",
            help="Décalage horizontal des étiquettes",
        )
        draw_endcaps = st.checkbox("Traits verticaux aux extrémités", value=False, help="Dessine des traits aux extrémités des barres")
        endcap_len = st.slider("Longueur end-caps (en Y)", 0.1, 1.0, 0.35, step=0.05, help="Longueur des traits de fin")
        show_today_line = st.checkbox("Ligne aujourd’hui", value=True, help="Affiche une ligne pour la date actuelle")

with tabs[2]:
    st.subheader("Annotations")
    ann_tabs = st.tabs(["Jalons", "Inputs", "Zones hachurées"])
    with ann_tabs[0]:
        st.subheader("Jalons")
        st.caption("🔖 Ajoutez des jalons pour marquer des étapes clés.")
        milestones_vlines = st.checkbox("Lignes verticales jalons", value=False, help="Trace des lignes verticales pour chaque jalon")
        ms_vline_titles = st.checkbox(
            "Titres sur lignes jalons",
            value=False,
            help="Affiche le titre du jalon (sur la ligne verticale ou directement sur l’axe s’il n’y a pas de ligne)",
        )

        ms_title_mode = "vline"
        ms_vline_title_pos = "Dans le graphique"
        ms_vline_title_y = 0.5
        ms_vline_title_offset = 0.02
        ms_inline_title_dx = 1.0
        ms_inline_title_dy = 0.0
        ms_vline_title_alpha = 0.5

        if ms_vline_titles:
            if milestones_vlines:
                ms_title_mode = "vline"
                ms_vline_title_pos = st.radio(
                    "Placement titre jalon",
                    ["Dans le graphique", "Au-dessus", "En dessous"],
                    help="Position du titre le long de la ligne verticale",
                )
                if ms_vline_title_pos == "Dans le graphique":
                    ms_vline_title_y = st.slider(
                        "Position titre jalon (axes fraction)",
                        0.0,
                        1.0,
                        0.5,
                        step=0.05,
                        help="Position verticale du titre sur la ligne",
                    )
                    ms_vline_title_offset = 0.02
                else:
                    ms_vline_title_y = 0.5
                    ms_vline_title_offset = st.slider(
                        "Décalage vertical titre jalon",
                        0.0,
                        0.2,
                        0.02,
                        step=0.01,
                        help="Décalage hors du graphique",
                    )
                ms_vline_title_alpha = st.slider(
                    "Transparence titre jalon",
                    0.0,
                    1.0,
                    0.5,
                    step=0.05,
                    help="Opacité du titre sur la ligne",
                )
            else:
                ms_title_mode = "inline"
                ms_inline_title_dx = st.slider(
                    "Décalage horizontal titre jalon (jours)",
                    0.0,
                    10.0,
                    1.5,
                    step=0.1,
                    help="Décalage du titre vers la droite après la date du jalon",
                )
                ms_inline_title_dy = st.slider(
                    "Décalage vertical titre jalon (axes fraction)",
                    -0.2,
                    0.2,
                    0.0,
                    step=0.01,
                    help="Ajuste la position verticale du titre par rapport au jalon",
                )
                ms_vline_title_alpha = st.slider(
                    "Transparence titre jalon",
                    0.0,
                    1.0,
                    0.5,
                    step=0.05,
                    help="Opacité du titre du jalon",
                )
        ms_markersize = st.slider("Taille marqueurs jalons", 8, 36, 16, step=1, help="Taille des marqueurs de jalons")
        ms_offset = st.slider("Offset vertical jalons (axes fraction)", 0.0, 0.1, 0.04, step=0.005, help="Décalage vertical des jalons")
        anti_overlap = st.checkbox("Anti-chevauchement (titres/jalons/inputs)", value=False, help="Évite le chevauchement des éléments")
        ms_alt_extra = st.slider("Alternance offset jalons (ajout max)", 0.0, 0.08, 0.03, step=0.005, help="Amplitude de l'alternance de l'offset")
        show_ms_dates = st.checkbox("Afficher date jalon", value=True, help="Affiche la date près du jalon")
        ms_date_pos = (
            st.selectbox("Position date jalon", ["Opposé à la flèche", "Près de la pointe"], help="Position du texte de date")
            if show_ms_dates
            else None
        )
        ms_date_offset = (
            st.slider("Offset vertical date jalon (axes fraction)", 0.0, 0.1, 0.02, step=0.005, help="Décalage vertical du texte de date")
            if show_ms_dates
            else 0.02
        )
        ms_date_fmt = (
            st.text_input("Format date jalon", "%d/%m", help="Format d'affichage de la date du jalon") if show_ms_dates else "%d/%m"
        )
    with ann_tabs[1]:
        st.subheader("Inputs")
        st.caption("📥 Renseignez des inputs pour contextualiser la timeline.")
        show_inputs = st.checkbox("Afficher inputs", value=True, help="Active l'affichage des inputs")
        inputs_position = st.selectbox(
            "Position des inputs",
            ["Haut (flèches descendantes)", "Bas (flèches montantes)"],
            index=0,
            help="Position des marqueurs d'inputs",
        )
        inputs_alt_extra = st.slider(
            "Alternance hauteur inputs (ajout max)", 0.0, 1.0, 0.4, step=0.05, help="Amplitude de l'alternance verticale"
        )
        inputs_top_margin = (
            st.slider("Marge verticale haut (Y)", 0.0, 2.0, 1.0, step=0.1, help="Espace au-dessus des inputs")
            if inputs_position == "Haut (flèches descendantes)"
            else 1.0
        )
        inputs_bottom_margin = (
            st.slider("Marge verticale bas (Y)", 0.0, 2.0, 1.0, step=0.1, help="Espace en dessous des inputs")
            if inputs_position == "Bas (flèches montantes)"
            else 1.0
        )
        input_arrow_len = st.slider("Longueur flèche input (Y)", 0.2, 2.0, 0.8, step=0.1, help="Longueur des flèches des inputs")
        show_input_dates = st.checkbox("Afficher date input", value=False, help="Affiche la date près de l'input")
        _input_fmt_options = {
            "31/12": "%d/%m",
            "12/31": "%m/%d",
            "Dec 31": "%b %d",
        }
        if show_input_dates:
            _input_fmt_label = st.selectbox(
                "Format date input",
                list(_input_fmt_options.keys()) + ["Personnalisé…"],
                index=0,
                help="Format d'affichage de la date",
            )
            input_date_fmt = (
                st.text_input(
                    "Format date input",
                    "%d/%m",
                    help="Format d'affichage de la date",
                )
                if _input_fmt_label == "Personnalisé…"
                else _input_fmt_options[_input_fmt_label]
            )
        else:
            input_date_fmt = "%d/%m"
        input_date_offset = (
            st.slider("Décalage vertical date input (Y)", 0.0, 1.0, 0.15, step=0.05, help="Décalage vertical de la date")
            if show_input_dates
            else 0.15
        )
        input_date_rot = (
            st.slider("Rotation date input", 0, 90, 90, step=5, help="Angle de la date affichée") if show_input_dates else 90
        )
        input_titles_in_legend = st.checkbox("Titres inputs en légende", value=False, help="Ajoute les inputs à la légende")
        input_title_date_split = st.checkbox("Date & titre de part et d'autre", value=False, help="Sépare la date et le titre")
        input_lr_offset = (
            st.slider("Décalage horizontal date/titre (jours)", 0.0, 5.0, 0.5, step=0.1, help="Décalage horizontal entre date et titre")
            if input_title_date_split
            else 0.5
        )

    with ann_tabs[2]:
        st.subheader("Zones hachurées")
        st.caption("🏖️ Mettez en évidence des périodes (congés, indisponibilités) avec une zone hachurée sur toute la hauteur du planning.")
        leave_zone_enabled = st.checkbox(
            "Afficher une zone hachurée",
            value=False,
            help="Ajoute une zone verticale couvrant toute la hauteur du graphique entre deux dates.",
        )

        _normalize_leave_zones_state()
        if leave_zone_enabled:
            st.caption(
                "Ajoutez plusieurs plages de dates à hachurer. Par défaut, chaque zone s'intitule 'Congés'."
            )
            leave_zones_df = st.data_editor(
                st.session_state.leave_zones,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "start": st.column_config.DateColumn("Début", default=date.today()),
                    "end": st.column_config.DateColumn("Fin", default=date.today()),
                    "label": st.column_config.TextColumn("Titre", default="Congés", width="medium"),
                },
                hide_index=True,
                key="leave_zones_editor",
            )
            if leave_zones_df is not None:
                st.session_state.leave_zones = leave_zones_df
        leave_zone_color = st.color_picker(
            "Couleur de la zone",
            "#B0B0B0",
            help="Définit la couleur de remplissage de la zone.",
        )
        leave_zone_edge_color = st.color_picker(
            "Couleur du bord",
            "#707070",
            help="Couleur du contour de la zone hachurée.",
        )
        leave_zone_alpha = st.slider(
            "Transparence",
            0.0,
            1.0,
            0.25,
            step=0.05,
            help="Opacité de la zone (0 = transparent, 1 = opaque).",
        )
        leave_zone_hatch = st.selectbox(
            "Motif de hachures",
            ["//", "\\\\", "xx", "..", "++", ""],
            index=0,
            help="Choisir le motif de hachures.",
        )
        leave_zone_label_color = st.color_picker(
            "Couleur du texte",
            "#000000",
            help="Couleur du texte affiché sur la zone.",
        )
        leave_zone_label_y = st.slider(
            "Position verticale du texte",
            0.0,
            1.0,
            0.9,
            step=0.05,
            help="Position verticale du texte (en fraction de l'axe Y).",
        )

with tabs[3]:
    st.subheader("Dépendances")
    show_dependencies = st.checkbox("Afficher dépendances", value=False, help="Affiche les liens entre les tâches")
    if show_dependencies:
        dep_conn_type = st.selectbox("Type liaison dépendances", ["Courbe", "Orthogonale"], index=0, help="Style de connexion des dépendances")
        if dep_conn_type == "Courbe":
            dep_arrow_rad = st.slider("Courbure flèches dépendances", -1.0, 1.0, 0.3, step=0.05, help="Ajuste la courbure des flèches")
            dep_connstyle = f"arc3,rad={dep_arrow_rad}"
        else:
            dep_connstyle = "angle3,angleA=0,angleB=90"
        dep_arrow_style = st.selectbox("Forme flèches dépendances", ["-|>", "->", "-["], index=0, help="Style de la tête des flèches")
        dep_arrow_color = st.color_picker("Couleur flèches dépendances", "#000000", help="Couleur des flèches de dépendance")
        dep_arrow_alpha = st.slider("Transparence flèches dépendances", 0.0, 1.0, 1.0, step=0.05, help="Opacité des flèches")
        dep_arrow_lw = st.slider("Épaisseur flèches dépendances", 0.5, 5.0, 1.0, step=0.5, help="Épaisseur du trait des flèches")
        dep_arrow_ms = st.slider("Taille tête flèches dépendances", 5, 30, 10, help="Taille de la pointe des flèches")
    else:
        dep_connstyle = ""
        dep_arrow_style = "-|>"
        dep_arrow_color = "black"
        dep_arrow_alpha = 1.0
        dep_arrow_lw = 1.0
        dep_arrow_ms = 10

st.sidebar.markdown("---")
fig_w = st.sidebar.slider("Largeur figure", 6, 20, 16, help="Largeur de la figure Matplotlib")
fig_h = st.sidebar.slider("Hauteur figure", 4, 12, 6, help="Hauteur de la figure Matplotlib")
st.sidebar.caption("💡 Télécharge le PNG en bas de page.")

# ============================== Meta Infos ==============================
st.sidebar.divider()
st.sidebar.caption("👤 Auteur : Ibrahim Bitar")
st.sidebar.caption("🏷️ Version : v1.1.0")
st.sidebar.caption("📅 Release : 04/09/2025")

# ============================== Data loading ==============================

combined_meta: dict[str, pd.DataFrame] = {}

try:
    if mode_data == "Démo":
        df_all = demo_dataframe()
        df_inputs = pd.DataFrame({"date": ["2025-09-10", "2025-10-05"], "label": ["Données A1 - EDV", "Données A2 - EDV"]})
        df_holidays = pd.DataFrame(columns=["date"])
    elif mode_data == "Fichier combiné":
        if uploaded_combined is None:
            st.info("➡️ Importez un fichier combiné ou passez en mode Démo.")
            st.stop()
        df_all, combined_meta = parse_combined_file(uploaded_combined)
        if uploaded_inputs:
            df_inputs = parse_inputs_file(uploaded_inputs)
        else:
            df_inputs = combined_meta.get("inputs", pd.DataFrame(columns=["date", "label"]))

        if combined_meta.get("leave_zones") is not None:
            st.session_state.leave_zones = combined_meta["leave_zones"].copy()

        if uploaded_holidays:
            df_holidays = parse_holidays_file(uploaded_holidays)
        else:
            df_holidays = combined_meta.get("holidays", pd.DataFrame(columns=["date"]))

        edit_csv = st.sidebar.checkbox(
            "Éditer les données importées",
            value=False,
            help="Permet de modifier les données après import",
        )
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

            with expander_search("Inputs (optionnel) — date, label", ["inputs"], default=False):
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
                    leave_zones_export = _normalize_leave_zones_dataframe(
                        st.session_state.get(
                            "leave_zones", pd.DataFrame(columns=["start", "end", "label"])
                        )
                    )
                    leave_zones_export.to_excel(writer, sheet_name="leave_zones", index=False)
                    holidays_export = _normalize_holidays_dataframe(df_holidays)
                    holidays_export.to_excel(writer, sheet_name="jours_feries", index=False)
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

    elif mode_data == "Édition (sans fichier)":
        _init_editor_state()
        st.subheader("✍️ Édition des données (sans fichier)")

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

        with expander_search("Inputs (optionnel) — date, label", ["inputs"], default=False):
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
        df_inputs = st.session_state.editor_inputs if not st.session_state.editor_inputs.empty else pd.DataFrame(columns=["date", "label"])
        df_holidays = pd.DataFrame(columns=["date"])

    else:  # Fichiers séparés
        if uploaded_acts is None or uploaded_miles is None:
            st.info("➡️ Importez les fichiers activities et milestones, ou passez en mode Démo.")
            st.stop()
        df_act = parse_activities_file(uploaded_acts)
        df_ms  = parse_milestones_file(uploaded_miles)
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
        df_inputs = parse_inputs_file(uploaded_inputs) if uploaded_inputs else pd.DataFrame(columns=["date", "label"])
        df_holidays = parse_holidays_file(uploaded_holidays) if uploaded_holidays else pd.DataFrame(columns=["date"])
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

# Unique groups & default colors
group_palette = plt.get_cmap("tab20").colors
unique_groups = [g for g in df_tasks["group"].dropna().unique() if g]
if "group_colors" not in st.session_state:
    st.session_state.group_colors = {}
for i, g in enumerate(unique_groups):
    default = mcolors.to_hex(group_palette[i % len(group_palette)])
    st.session_state.group_colors.setdefault(g, default)

with tabs[1]:
    with expander_search("Couleurs des groupes", ["couleur", "groupe"]):
        for i, g in enumerate(unique_groups):
            color_key = f"group_color_{g}".replace(" ", "_")
            picked = st.color_picker(g, st.session_state.group_colors.get(g, "#000000"), key=color_key)
            st.session_state.group_colors[g] = picked

group_colors = {g: st.session_state.group_colors.get(g, "gray") for g in unique_groups}

if not df_inputs.empty:
    df_inputs["date"] = pd.to_datetime(df_inputs["date"]).dt.date
if not df_holidays.empty:
    df_holidays["date"] = pd.to_datetime(df_holidays["date"]).dt.date

if leave_zone_enabled:
    leave_zones_df = st.session_state.get("leave_zones", pd.DataFrame())
else:
    leave_zones_df = pd.DataFrame(columns=["start", "end", "label"])

leave_zones = []
if leave_zone_enabled and not leave_zones_df.empty:
    df_tmp = leave_zones_df.copy()
    for col in ("start", "end"):
        if col in df_tmp:
            df_tmp[col] = pd.to_datetime(df_tmp[col], errors="coerce").dt.date
    for row in df_tmp.itertuples(index=False):
        start = getattr(row, "start", None)
        end = getattr(row, "end", None)
        if start and end:
            if start > end:
                start, end = end, start
            label_raw = getattr(row, "label", "")
            if isinstance(label_raw, str):
                label = label_raw.strip() or "Congés"
            elif pd.isna(label_raw):
                label = "Congés"
            else:
                label = str(label_raw)
            leave_zones.append({"start": start, "end": end, "label": label})

leave_zone_holidays = leave_zones_to_holidays(leave_zones)
holiday_arrays: list[np.ndarray] = []

if leave_zone_holidays is not None and len(leave_zone_holidays) > 0:
    holiday_arrays.append(leave_zone_holidays)

if not df_holidays.empty:
    try:
        official_np = np.array(df_holidays["date"], dtype="datetime64[D]")
    except Exception:
        official_np = pd.to_datetime(df_holidays["date"], errors="coerce").dropna().to_numpy(dtype="datetime64[D]")
    if official_np.size > 0:
        holiday_arrays.append(official_np)

working_day_calendar = None
if holiday_arrays:
    combined_holidays = np.unique(np.concatenate(holiday_arrays))
    if combined_holidays.size > 0:
        working_day_calendar = np.busdaycalendar(holidays=combined_holidays)

# Pré-calcul des durées en jours travaillés pour les activités
if not df_tasks.empty:
    df_tasks["duration_workdays"] = [
        business_days_between(
            row.start,
            row.end,
            inclusive=inclusive_duration,
            busdaycal=working_day_calendar,
        )
        for row in df_tasks.itertuples(index=False)
    ]
else:
    df_tasks["duration_workdays"] = pd.Series(dtype="int64")

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

if leave_zones:
    zone_starts = [z["start"] for z in leave_zones]
    zone_ends = [z["end"] for z in leave_zones]
    if zone_starts:
        zone_min = min(zone_starts)
        data_min = zone_min if (data_min is None or zone_min < data_min) else data_min
    if zone_ends:
        zone_max = max(zone_ends)
        data_max = zone_max if (data_max is None or zone_max > data_max) else data_max

# fallback si toujours None → fenêtre centrée sur aujourd’hui
if data_min is None or data_max is None:
    today = pd.Timestamp.today().date()
    data_min = today
    data_max = today

# overrides utilisateur
def _parse_date_or(default, value):
    try:
        return pd.to_datetime(value if value else default)
    except Exception:
        return pd.to_datetime(default)

x_min_user = _parse_date_or(data_min, x_min_val)
x_max_user = _parse_date_or(data_max, x_max_val)

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
default_color = "gray"
legend_groups = []
legend_other = []

# Bars + labels
task_positions = {}
for i, row in enumerate(df_tasks.itertuples(index=False), start=1):
    start_num = mdates.date2num(row.start)
    end_num   = mdates.date2num(row.end)
    duration_num = end_num - start_num
    duration_days = getattr(row, "duration_workdays", None)
    if duration_days is None or pd.isna(duration_days):
        duration_days = business_days_between(
            row.start,
            row.end,
            inclusive=inclusive_duration,
            busdaycal=working_day_calendar,
        )
    duration_weeks = duration_days / 5.0 if duration_days else 0.0
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
        if title_align_start:
            x_title = start_num
            ha_title = "left"
        else:
            x_title = start_num + duration_num / 2.0
            ha_title = "center"
        fs, _ = _fit_text_in_span(
            ax, row.title,
            x_center=x_title,
            y=y_title, x_left=start_num, x_right=end_num,
            max_font_size=title_max_fs, min_font_size=title_min_fs, padding_px=title_padding_px,
            zorder=6, ha=ha_title, va="center", clip_on=False
        )
        if fs is None:
            ax.text(
                x_title,
                y_title,
                row.title,
                va="center",
                ha=ha_title,
                fontsize=title_min_fs,
                zorder=6,
                clip_on=False,
            )

    # duration inside bar (unit-aware)
    if dur_in_bar and duration_num > 0:
        if dur_unit == "Jours":
            dur_txt = dur_fmt.format(d=duration_days)
        else:
            dur_txt = dur_fmt.format(w=duration_weeks)
        ax.text(
            start_num + duration_num / 2.0,
            i,
            dur_txt,
            va="center",
            ha="center",
            fontsize=dur_font,
            zorder=5,
            clip_on=True,
        )

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
    legend_groups.append(Line2D([0], [0], color=c, linewidth=6, label=g))

# Milestones at y=0
palette = plt.get_cmap("tab10").colors
blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)

for k, row in enumerate(df_ms.itertuples(index=False)):
    x_ms = mdates.date2num(row.start)
    # alternating vertical offset (axes fraction)
    extra = (ms_alt_extra if (anti_overlap and k % 2 == 1) else 0.0)
    y_af = ms_offset + extra

    # vline
    if milestones_vlines:
        ax.axvline(x=x_ms, linestyle=":", alpha=0.3, zorder=3)
    if ms_vline_titles:
        if ms_title_mode == "vline":
            if ms_vline_title_pos == "Au-dessus":
                y_text = 1 + ms_vline_title_offset
                va = "bottom"
            elif ms_vline_title_pos == "En dessous":
                y_text = -ms_vline_title_offset
                va = "top"
            else:
                y_text = ms_vline_title_y
                va = "center"

            ax.text(
                x_ms,
                y_text,
                row.title,
                rotation=90,
                ha="center",
                va=va,
                rotation_mode="anchor",  # pour ancrer le début du texte
                alpha=ms_vline_title_alpha,
                transform=blended,
                zorder=3,
                clip_on=False,
            )
        else:
            ax.text(
                x_ms + ms_inline_title_dx,
                y_af + ms_inline_title_dy,
                row.title,
                rotation=90,
                rotation_mode="anchor",
                ha="left",
                va="center",
                alpha=ms_vline_title_alpha,
                transform=blended,
                zorder=7,
                clip_on=False,
            )
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

    legend_other.append(
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
            legend_other.append(Line2D([0],[0], marker=marker, linestyle="None",
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

# Create separate legends for milestones/other markers and activity groups
if legend_other or legend_groups:
    legend_spacing = 0.02
    if legend_loc == "outside right":
        x_anchor = 1.02
        if legend_other:
            legend_jalons = ax.legend(
                handles=legend_other,
                ncol=1,
                loc="upper left",
                bbox_to_anchor=(x_anchor, 1.0),
                frameon=False,
            )
            ax.add_artist(legend_jalons)
            fig.canvas.draw()
            bbox = legend_jalons.get_window_extent(renderer=fig.canvas.get_renderer())
            width = ax.transAxes.inverted().transform_bbox(bbox).width
            x_anchor += width + legend_spacing
        if legend_groups:
            legend_activites = ax.legend(
                handles=legend_groups,
                ncol=1,
                loc="upper left",
                bbox_to_anchor=(x_anchor, 1.0),
                frameon=False,
            )
    elif legend_loc == "outside left":
        x_anchor = -0.02
        if legend_other:
            legend_jalons = ax.legend(
                handles=legend_other,
                ncol=1,
                loc="upper right",
                bbox_to_anchor=(x_anchor, 1.0),
                frameon=False,
            )
            fig.canvas.draw()
            bbox = legend_jalons.get_window_extent(renderer=fig.canvas.get_renderer())
            width = ax.transAxes.inverted().transform_bbox(bbox).width
            legend_jalons.set_bbox_to_anchor((x_anchor - width - legend_spacing, 1.0))
            ax.add_artist(legend_jalons)
        if legend_groups:
            legend_activites = ax.legend(
                handles=legend_groups,
                ncol=1,
                loc="upper right",
                bbox_to_anchor=(x_anchor, 1.0),
                frameon=False,
            )
    elif legend_loc == "outside bottom":
        y_anchor = -0.02
        x_anchor = 0.5
        if legend_other:
            legend_jalons = ax.legend(
                handles=legend_other,
                ncol=1,
                loc="upper center",
                bbox_to_anchor=(x_anchor, y_anchor),
                frameon=False,
            )
            ax.add_artist(legend_jalons)
            fig.canvas.draw()
            bbox = legend_jalons.get_window_extent(renderer=fig.canvas.get_renderer())
            width = ax.transAxes.inverted().transform_bbox(bbox).width
            legend_jalons.set_bbox_to_anchor((x_anchor - width / 2 - legend_spacing / 2, y_anchor))
            x_anchor = x_anchor + width / 2 + legend_spacing / 2
        if legend_groups:
            legend_activites = ax.legend(
                handles=legend_groups,
                ncol=1,
                loc="upper center",
                bbox_to_anchor=(x_anchor, y_anchor),
                frameon=False,
            )
    elif legend_loc == "outside top":
        y_anchor = 1.02
        x_anchor = 0.5
        if legend_other:
            legend_jalons = ax.legend(
                handles=legend_other,
                ncol=1,
                loc="lower center",
                bbox_to_anchor=(x_anchor, y_anchor),
                frameon=False,
            )
            ax.add_artist(legend_jalons)
            fig.canvas.draw()
            bbox = legend_jalons.get_window_extent(renderer=fig.canvas.get_renderer())
            width = ax.transAxes.inverted().transform_bbox(bbox).width
            legend_jalons.set_bbox_to_anchor((x_anchor - width / 2 - legend_spacing / 2, y_anchor))
            x_anchor = x_anchor + width / 2 + legend_spacing / 2
        if legend_groups:
            legend_activites = ax.legend(
                handles=legend_groups,
                ncol=1,
                loc="lower center",
                bbox_to_anchor=(x_anchor, y_anchor),
                frameon=False,
            )
    else:
        y_anchor = 1.0 if "upper" in legend_loc else (0.0 if "lower" in legend_loc else 0.5)
        if legend_other:
            legend_jalons = ax.legend(
                handles=legend_other,
                ncol=1,
                loc=legend_loc,
                bbox_to_anchor=(-0.1, y_anchor),
                frameon=False,
            )
            ax.add_artist(legend_jalons)
            fig.canvas.draw()
            bbox = legend_jalons.get_window_extent(renderer=fig.canvas.get_renderer())
            width = ax.transAxes.inverted().transform_bbox(bbox).width
            legend_jalons.set_bbox_to_anchor((-0.1 - width - legend_spacing, y_anchor))
        if legend_groups:
            legend_activites = ax.legend(
                handles=legend_groups,
                ncol=1,
                loc=legend_loc,
                bbox_to_anchor=(0.1, y_anchor),
                frameon=False,
            )

# X axis limits and ticks
ax.set_xlim(x_min_num, x_max_num)
if highlight_weekends or not df_holidays.empty:
    holidays = set(df_holidays["date"]) if not df_holidays.empty else set()
    for day_num in range(math.floor(x_min_num), math.ceil(x_max_num)):
        day = mdates.num2date(day_num).date()
        if (highlight_weekends and day.weekday() in {5, 6}) or (day in holidays):
            ax.axvspan(day_num, day_num + 1, color='0.9', zorder=1)
if leave_zones:
    blended_zone = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for zone in leave_zones:
        zone_start_num = mdates.date2num(zone["start"])
        zone_end_num = mdates.date2num(zone["end"] + timedelta(days=1))
        if zone_start_num > zone_end_num:
            zone_start_num, zone_end_num = zone_end_num, zone_start_num
        zone_patch = ax.axvspan(
            zone_start_num,
            zone_end_num,
            facecolor=leave_zone_color,
            edgecolor=leave_zone_edge_color,
            alpha=leave_zone_alpha,
            hatch=leave_zone_hatch,
            linewidth=1.0,
            zorder=1.4,
        )
        label_text = zone.get("label", "Congés")
        if label_text:
            ax.text(
                zone_start_num + (zone_end_num - zone_start_num) / 2.0,
                leave_zone_label_y,
                label_text,
                ha="center",
                va="center",
                color=leave_zone_label_color,
                fontsize=12,
                transform=blended_zone,
                zorder=zone_patch.get_zorder() + 0.1,
                clip_on=False,
            )
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
if show_ax_arrow:
    ax.annotate(
        '',
        xy=(x_max_num, 0.0), xycoords=('data', 'axes fraction'),
        xytext=(x_min_num, 0.0), textcoords=('data', 'axes fraction'),
        arrowprops=dict(
            arrowstyle='-|>',
            color=ax_arrow_color,
            lw=ax_arrow_lw,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=ax_arrow_scale,
        ),
        zorder=6,
    )

ax.set_xlabel("")
ax.set_ylabel("")
title_pad = 12 if planning_title_show_today else 6
ax.set_title(
    planning_title,
    fontsize=planning_title_size,
    color=planning_title_color,
    pad=title_pad,
)

if planning_title_show_today:
    today_label = datetime.now().strftime(planning_title_date_format)
    ax.text(
        0.5,
        1.02,
        today_label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=planning_title_subtitle_size,
        color=planning_title_color,
        fontstyle="italic",
    )

st.pyplot(fig, use_container_width=True)

if show_assumptions_note:
    note = assumptions_note_text.strip()
    if note:
        note_lines = [escape(line) for line in note.splitlines() if line.strip()]
        if note_lines:
            note_html = "<br>".join(note_lines)
            st.markdown(
                (
                    "<div style='font-size:0.85rem; color:#444; margin-top:0.5rem;'>"
                    "<strong>Hypothèses du planning :</strong><br>"
                    f"{note_html}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

if not df_tasks.empty and "duration_workdays" in df_tasks.columns:
    with st.expander("📋 Durées des activités (jours travaillés)"):
        duration_table = df_tasks[["id", "title", "start", "end", "duration_workdays"]].copy()
        duration_table = duration_table.rename(
            columns={"id": "ID", "title": "Titre", "start": "Début", "end": "Fin", "duration_workdays": "Durée (jours travaillés)"}
        )
        st.dataframe(duration_table, hide_index=True, use_container_width=True)

# ============================== Download PNG ==============================

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
timestamp_label = datetime.now().strftime("%Y%m%d-%H%M%S")
png_filename = f"planning_{timestamp_label}.png"
st.download_button(
    label="⬇️ Télécharger le PNG",
    data=buf.getvalue(),
    file_name=png_filename,
    mime="image/png",
)


# ============================== Help / Schemas ==============================

