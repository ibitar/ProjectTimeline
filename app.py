import json
import re
from string import Template
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go


st.set_page_config(page_title="Interactive Gantt", layout="wide")


# ---------- Data management ----------
def load_data():
    """Load tasks into session state."""
    if "tasks" not in st.session_state:
        st.session_state["tasks"] = pd.read_csv(
            "example_tasks.csv", parse_dates=["start", "end"]
        )
    return st.session_state["tasks"]


# ---------- Figure ----------
def make_fig(df: pd.DataFrame) -> go.Figure:
    """Create a Plotly figure with draggable task bars."""
    fig = go.Figure()
    for idx, row in df.iterrows():
        fig.add_shape(
            type="rect",
            x0=row["start"],
            x1=row["end"],
            y0=idx - 0.4,
            y1=idx + 0.4,
            line=dict(color="royalblue"),
            fillcolor="royalblue",
            opacity=0.6,
        )
        # task title
        mid = row["start"] + (row["end"] - row["start"]) / 2
        fig.add_trace(
            go.Scatter(
                x=[mid],
                y=[idx],
                text=row["title"],
                mode="text",
                showlegend=False,
            )
        )
    fig.update_yaxes(
        tickvals=list(range(len(df))),
        ticktext=df["id"],
        autorange="reversed",
        title="",
    )
    fig.update_xaxes(type="date", title="")
    fig.update_layout(
        height=400,
        dragmode="pan",
        editable=True,
        margin=dict(l=20, r=20, t=30, b=20),
        shapes=list(fig.layout.shapes),
        showlegend=False,
    )
    return fig


# ---------- UI ----------
st.title("Interactive Gantt")

df = load_data()
fig = make_fig(df)
fig_json = fig.to_json()

html_template = Template(
    """
<div id='gantt'></div>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
<script>
const plotly_data = $fig_json;
var gd = document.getElementById('gantt');
Plotly.newPlot(gd, plotly_data.data, plotly_data.layout);
gd.on('plotly_click', function(e){
  const msg = {"event": "click", "point": e.points[0]};
  window.parent.postMessage({"isStreamlitMessage": true, "type": "streamlit:setComponentValue", "value": JSON.stringify(msg)}, '*');
});
gd.on('plotly_relayout', function(e){
  const msg = {"event": "relayout", "data": e};
  window.parent.postMessage({"isStreamlitMessage": true, "type": "streamlit:setComponentValue", "value": JSON.stringify(msg)}, '*');
});
</script>
"""
)
html = html_template.substitute(fig_json=fig_json)

# Event capturing
result = components.html(html, height=600)

if result:
    info = json.loads(result)
    if info.get("event") == "relayout":
        relayout = info["data"]
        for key, val in relayout.items():
            m = re.match(r"shapes\[(\d+)\]\.(x0|x1|y0|y1)", key)
            if not m:
                continue
            idx = int(m.group(1))
            field = m.group(2)
            if field == "x0":
                df.at[idx, "start"] = pd.to_datetime(val)
            elif field == "x1":
                df.at[idx, "end"] = pd.to_datetime(val)
            elif field in {"y0", "y1"}:
                y0 = relayout.get(f"shapes[{idx}].y0", df.index[idx] - 0.4)
                y1 = relayout.get(f"shapes[{idx}].y1", df.index[idx] + 0.4)
                new_idx = round((y0 + y1) / 2)
                if 0 <= new_idx < len(df) and new_idx != idx:
                    df.iloc[[idx, new_idx]] = df.iloc[[new_idx, idx]].values
                    df.reset_index(drop=True, inplace=True)
        st.session_state["tasks"] = df
        fig = make_fig(df)

st.plotly_chart(fig, use_container_width=True)

# ---------- Export ----------
export_df = df.copy()
export_df["start"] = export_df["start"].dt.strftime("%Y-%m-%d")
export_df["end"] = export_df["end"].dt.strftime("%Y-%m-%d")
st.download_button(
    "💾 Exporter les tâches",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name="tasks_updated.csv",
    mime="text/csv",
)
