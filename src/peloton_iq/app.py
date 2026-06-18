"""
peloton_iq.app
~~~~~~~~~~~~~~
PelotonIQ Dash application.

Calls the FastAPI backend (run_api.py) via HTTP.
Agent lives in FastAPI — Dash is a pure UI layer.

Run (requires FastAPI running on port 8000 first):
    python scripts/run_api.py     # terminal 1
    python scripts/run_dash.py    # terminal 2
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
from dash import ALL, Dash, Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

from peloton_iq.config import COURSE_CLEAN_PATH
from peloton_iq.ingestion.gpx import load_elevation_profile, get_climb_annotations, find_gpx_path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("PELOTON_API_URL", "http://localhost:8000")


def query_api(query: str) -> dict:
    try:
        r = requests.post(f"{API_BASE}/api/query", json={"query": query}, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {
            "error":           "Cannot reach PelotonIQ API. Make sure run_api.py is running on port 8000.",
            "response":        "",
            "query_type":      "",
            "steps":           [],
            "elapsed_s":       0,
            "prediction_text": None,
            "race_context":    None,
        }
    except Exception as e:
        return {
            "error":           str(e),
            "response":        "",
            "query_type":      "",
            "steps":           [],
            "elapsed_s":       0,
            "prediction_text": None,
            "race_context":    None,
        }


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        return r.ok
    except Exception:
        return False



def fetch_results(race_name: str, year: int, stage=None, top_n: int = 10) -> dict:
    """Fetch actual race results from the FastAPI backend."""
    try:
        params = {"race_name": race_name, "year": year, "top_n": top_n}
        if stage:
            params["stage"] = stage
        r = requests.get(f"{API_BASE}/api/results", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"found": False, "results": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

C = {
    "bg":        "#0B0E14",
    "surface":   "#141820",
    "elevated":  "#1E2430",
    "border":    "#2A3140",
    "text":      "#E6EDF3",
    "secondary": "#8B949E",
    "accent":    "#F5C518",
    "accent_dim":"#7A6209",
    "red":       "#E8323C",
    "green":     "#1A9E5C",
    "blue":      "#1D6FA4",
}

FONT = "'Inter', 'Segoe UI', -apple-system, sans-serif"
MONO = "'JetBrains Mono', 'Fira Code', 'Consolas', monospace"

# ---------------------------------------------------------------------------
# Suggested queries
# ---------------------------------------------------------------------------

SUGGESTED = [
    ("TDF 2023 Stage 17 briefing",
     "It's before stage 17 of the 2023 Tour de France. Give me a full pre-race briefing."),
    ("Who won TDF Stage 17?",
     "Who won Tour de France Stage 17 in 2023?"),
    ("Paris-Roubaix course profile",
     "What makes Paris-Roubaix so different from other classics?"),
    ("Evenepoel 2023 season",
     "How did Remco Evenepoel perform across his seasons in our dataset?"),
    ("Best mountain riders",
     "Which riders have historically performed best on high mountain stages?"),
    ("Strade Bianche 2022 results",
     "Top 10 results of Strade Bianche 2022"),
]

# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_prediction_chart(prediction_text: str) -> go.Figure:
    if not prediction_text or "NO PREDICTION" in prediction_text:
        return _empty_chart("Run a predictive query to see win probabilities")

    pattern = re.compile(
        r"^\s+\d+\.\s+(.+?)\s{2,}win:\s*([\d.]+)%\s+podium\+:\s*([\d.]+)%",
        re.MULTILINE,
    )
    matches = pattern.findall(prediction_text)
    if not matches:
        return _empty_chart("No probability data found")

    riders    = [m[0].strip()[:28] for m in matches][:10]
    win_probs = [float(m[1]) for m in matches][:10]
    pod_probs = [float(m[2]) for m in matches][:10]

    bar_colors  = [C["accent"]] + [C["border"]] * (len(riders) - 1)
    text_colors = [C["bg"]]     + [C["secondary"]] * (len(riders) - 1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Win %",
        y=riders[::-1],
        x=win_probs[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        text=[f"{p:.1f}%" for p in win_probs[::-1]],
        textposition="inside",
        textfont=dict(family=MONO, size=11, color=text_colors[::-1]),
        hovertemplate="<b>%{y}</b><br>Win: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Podium+ %",
        y=riders[::-1],
        x=pod_probs[::-1],
        orientation="h",
        marker_color="rgba(245,197,24,0.15)",
        marker_line=dict(color=C["accent_dim"], width=1),
        hovertemplate="<b>%{y}</b><br>Podium+: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(family=FONT, color=C["text"], size=12),
        margin=dict(l=0, r=16, t=8, b=8),
        height=320,
        xaxis=dict(
            title=dict(text="Probability (%)", font=dict(size=11, color=C["secondary"])),
            tickfont=dict(family=MONO, size=10, color=C["secondary"]),
            gridcolor=C["border"],
            zerolinecolor=C["border"],
            range=[0, max(pod_probs) * 1.15 + 1],
        ),
        yaxis=dict(
            tickfont=dict(size=11, color=C["text"]),
            gridcolor="rgba(0,0,0,0)",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=10, color=C["secondary"]),
        ),
        hoverlabel=dict(bgcolor=C["elevated"], bordercolor=C["border"],
                        font=dict(family=FONT, size=12)),
    )
    return fig


def build_course_chart(race_name: str, year: int, stage: Optional[int]) -> go.Figure:
    # Build the full race name to look up GPX
    if stage:
        full_name = f"{year} {race_name} Stage {stage}"
    else:
        full_name = f"{year} {race_name}"

    # Try GPX first — real elevation profile
    gpx_df = load_elevation_profile(full_name)

    if gpx_df is not None and not gpx_df.empty:
        return _build_gpx_chart(gpx_df, full_name)

    # Fall back to aggregate stats from course_data_clean.csv
    return _build_stats_chart(race_name, year, stage)


def _build_gpx_chart(gpx_df: pd.DataFrame, label: str) -> go.Figure:
    """Real elevation profile from GPX data."""
    dist  = gpx_df["distance_km"]
    elev  = gpx_df["elevation_m"]
    grad  = gpx_df["gradient_pct"]

    # Colour gradient bars by steepness
    bar_colors = []
    for g in grad:
        if g >= 10:    bar_colors.append("rgba(232,50,60,0.8)")    # steep — red
        elif g >= 6:   bar_colors.append("rgba(245,197,24,0.8)")   # hard — yellow
        elif g >= 3:   bar_colors.append("rgba(29,111,164,0.6)")   # moderate — blue
        else:          bar_colors.append("rgba(42,49,64,0.4)")     # flat — muted

    fig = go.Figure()

    # Elevation fill
    fig.add_trace(go.Scatter(
        x=dist, y=elev,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(29,111,164,0.15)",
        line=dict(color=C["blue"], width=1.5),
        hovertemplate="<b>%{x:.1f} km</b><br>%{y:.0f}m<extra></extra>",
        name="Elevation",
    ))

    # Gradient overlay as coloured scatter dots
    fig.add_trace(go.Scatter(
        x=dist, y=elev,
        mode="markers",
        marker=dict(color=bar_colors, size=3, symbol="circle"),
        hovertemplate="<b>%{x:.1f} km</b><br>%{y:.0f}m  grad: %{text}<extra></extra>",
        text=[f"{g:+.1f}%" for g in grad],
        showlegend=False,
    ))

    # Key stats annotation
    stats = [
        f"↑ {elev.max() - elev.min():,.0f}m",
        f"▲ {elev.max():,.0f}m",
        f"⬤ {dist.max():.0f}km",
    ]
    fig.add_annotation(
        x=0.02, y=0.96, xref="paper", yref="paper",
        text="  ·  ".join(stats),
        showarrow=False,
        font=dict(family=MONO, size=11, color=C["accent"]),
        bgcolor="rgba(11,14,20,0.75)",
        borderpad=6,
    )

    # Annotate steepest climb peaks
    climbs = get_climb_annotations(gpx_df, min_gradient=7.0)
    for i, climb in enumerate(climbs[:3]):   # max 3 annotations
        fig.add_annotation(
            x=climb["peak_km"],
            y=climb["peak_elevation"],
            text=f"{climb['avg_gradient']:+.0f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=C["accent"],
            arrowwidth=1,
            font=dict(size=9, color=C["accent"], family=MONO),
            bgcolor="rgba(11,14,20,0.7)",
            borderpad=3,
            ay=-25,
        )

    fig.update_layout(
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(family=FONT, color=C["text"], size=12),
        margin=dict(l=0, r=16, t=16, b=8),
        height=240,
        showlegend=False,
        xaxis=dict(
            title=dict(text="Distance (km)", font=dict(size=11, color=C["secondary"])),
            tickfont=dict(family=MONO, size=10, color=C["secondary"]),
            gridcolor=C["border"],
            zerolinecolor=C["border"],
        ),
        yaxis=dict(
            title=dict(text="Elevation (m)", font=dict(size=11, color=C["secondary"])),
            tickfont=dict(family=MONO, size=10, color=C["secondary"]),
            gridcolor=C["border"],
            zerolinecolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor=C["elevated"], bordercolor=C["border"],
                        font=dict(family=FONT, size=12)),
    )
    return fig


def _build_stats_chart(race_name: str, year: int, stage: Optional[int]) -> go.Figure:
    """Fallback chart from aggregate stats when no GPX is available."""
    try:
        course_df = pd.read_csv(COURSE_CLEAN_PATH)
    except Exception:
        return _empty_chart("Course data unavailable")

    year_col = "Year_results" if "Year_results" in course_df.columns else "Year"
    mask = (
        course_df["Race Name"].str.contains(race_name, case=False, na=False) &
        (course_df[year_col] == year)
    )
    if stage:
        mask &= course_df["Race Name"].str.contains(f"Stage {stage}", na=False)

    rows = course_df[mask]
    if rows.empty:
        rows = course_df[course_df["Race Name"].str.contains(race_name, case=False, na=False)]
    if rows.empty:
        return _empty_chart(f"No course data for {race_name}")

    row  = rows.iloc[0]
    vg   = float(row.get("Vertical Gain", 0) or 0)
    he   = float(row.get("Highest Elevation", 0) or 0)
    le   = float(row.get("Lowest Elevation", 0) or 0)
    dist = float(row.get("Distance", 200) or 200)
    cob  = float(row.get("Cobblestones", 0) or 0)

    x = [0, dist * 0.3, dist * 0.75, dist]
    y = [le, le + vg * 0.35, he, le + (he - le) * 0.4]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        fill="tozeroy",
        fillcolor="rgba(29,111,164,0.18)",
        line=dict(color=C["blue"], width=2),
        hovertemplate="<b>%{x:.0f} km</b><br>~%{y:.0f}m<extra></extra>",
    ))

    if cob > 0:
        fig.add_vrect(
            x0=dist * 0.55, x1=dist * 0.55 + cob,
            fillcolor="rgba(245,197,24,0.12)",
            line=dict(color=C["accent"], width=1, dash="dot"),
            annotation_text=f"⬡ {cob:.0f}km cobbles",
            annotation_position="top left",
            annotation_font=dict(color=C["accent"], size=10, family=FONT),
        )

    stats = [f"↑ {vg:,.0f}m", f"▲ {he:,.0f}m", f"⬤ {dist:.0f}km"]
    fig.add_annotation(
        x=0.02, y=0.96, xref="paper", yref="paper",
        text="  ·  ".join(stats) + "  (estimated profile)",
        showarrow=False,
        font=dict(family=MONO, size=10, color=C["secondary"]),
        bgcolor="rgba(11,14,20,0.7)",
        borderpad=6,
    )

    fig.update_layout(
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
        font=dict(family=FONT, color=C["text"], size=12),
        margin=dict(l=0, r=16, t=16, b=8), height=240, showlegend=False,
        xaxis=dict(
            title=dict(text="Distance (km)", font=dict(size=11, color=C["secondary"])),
            tickfont=dict(family=MONO, size=10, color=C["secondary"]),
            gridcolor=C["border"], zerolinecolor=C["border"],
        ),
        yaxis=dict(
            title=dict(text="Elevation (m)", font=dict(size=11, color=C["secondary"])),
            tickfont=dict(family=MONO, size=10, color=C["secondary"]),
            gridcolor=C["border"], zerolinecolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor=C["elevated"], bordercolor=C["border"],
                        font=dict(family=FONT, size=12)),
    )
    return fig


def _empty_chart(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=message, showarrow=False,
        font=dict(family=FONT, size=13, color=C["secondary"]),
    )
    fig.update_layout(
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
        margin=dict(l=16, r=16, t=16, b=16), height=200,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _label(text: str) -> html.Div:
    return html.Div(text, style={
        "fontFamily": MONO, "fontSize": "10px",
        "letterSpacing": "0.12em", "textTransform": "uppercase",
        "color": C["secondary"], "marginBottom": "6px",
    })


def _section_card(*children, style=None) -> html.Div:
    base = {
        "background": C["surface"], "border": f"1px solid {C['border']}",
        "borderRadius": "8px", "padding": "20px 24px", "marginBottom": "16px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def _badge(text: str, color: str = None) -> html.Span:
    color = color or C["secondary"]
    return html.Span(text, style={
        "fontFamily": MONO, "fontSize": "10px", "letterSpacing": "0.08em",
        "textTransform": "uppercase", "color": color,
        "background": "rgba(255,255,255,0.05)", "border": f"1px solid {C['border']}",
        "borderRadius": "3px", "padding": "2px 7px", "marginRight": "6px",
    })


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = Dash(__name__, title="PelotonIQ — Race Intelligence",
           suppress_callback_exceptions=True)

app.layout = html.Div([

    dcc.Store(id="agent-result-store"),
    dcc.Store(id="prediction-text-store"),
    dcc.Store(id="race-context-store"),
    dcc.Store(id="results-store"),
    dcc.Interval(id="health-interval", interval=15000, n_intervals=0),

    # Header
    html.Div([
        html.Div([
            html.Span("PELOTON", style={
                "fontFamily": "'Barlow Condensed', 'Arial Narrow', sans-serif",
                "fontWeight": "800", "fontSize": "22px",
                "letterSpacing": "0.06em", "color": C["text"],
            }),
            html.Span("IQ", style={
                "fontFamily": "'Barlow Condensed', 'Arial Narrow', sans-serif",
                "fontWeight": "800", "fontSize": "22px",
                "letterSpacing": "0.06em", "color": C["accent"],
            }),
            html.Span("RACE INTELLIGENCE", style={
                "fontFamily": MONO, "fontSize": "9px", "letterSpacing": "0.18em",
                "color": C["secondary"], "marginLeft": "12px", "verticalAlign": "middle",
            }),
        ], style={"display": "flex", "alignItems": "baseline"}),

        html.Div([
            html.Span("UCI WorldTour · 2017–2023", style={
                "fontFamily": MONO, "fontSize": "10px",
                "color": C["secondary"], "marginRight": "12px",
            }),
            html.Div(id="status-indicator", style={
                "width": "8px", "height": "8px",
                "borderRadius": "50%", "background": C["border"],
                "display": "inline-block",
            }),
        ], style={"display": "flex", "alignItems": "center"}),

    ], style={
        "background": C["bg"], "borderBottom": f"1px solid {C['border']}",
        "padding": "0 28px", "height": "52px",
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "position": "sticky", "top": "0", "zIndex": "100",
    }),

    # Body
    html.Div([

        # Left — query + response
        html.Div([
            _section_card(
                _label("Intelligence query"),
                dcc.Textarea(
                    id="query-input", value="", maxLength=500,
                    placeholder="Ask about race history, course profiles, rider form, or request a pre-race briefing…",
                    style={
                        "width": "100%", "minHeight": "80px",
                        "background": C["elevated"], "border": f"1px solid {C['border']}",
                        "borderRadius": "6px", "color": C["text"],
                        "fontFamily": FONT, "fontSize": "14px",
                        "lineHeight": "1.6", "padding": "12px 14px",
                        "resize": "none", "outline": "none", "marginBottom": "12px",
                    },
                ),
                html.Div([
                    html.Button(
                        label,
                        id={"type": "suggestion-btn", "index": i},
                        n_clicks=0,
                        **{"data-query": query},
                        style={
                            "background": "transparent", "border": f"1px solid {C['border']}",
                            "borderRadius": "20px", "color": C["secondary"],
                            "fontFamily": FONT, "fontSize": "11px",
                            "padding": "4px 12px", "cursor": "pointer",
                            "marginRight": "6px", "marginBottom": "6px",
                        },
                    )
                    for i, (label, query) in enumerate(SUGGESTED)
                ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "12px"}),

                html.Div([
                    html.Button("▶  Analyse", id="run-btn", n_clicks=0, style={
                        "background": C["accent"], "border": "none",
                        "borderRadius": "6px", "color": C["bg"],
                        "fontFamily": FONT, "fontWeight": "700",
                        "fontSize": "13px", "padding": "9px 22px", "cursor": "pointer",
                    }),
                    html.Div(id="query-meta", style={
                        "display": "flex", "alignItems": "center",
                        "gap": "8px", "marginLeft": "16px",
                    }),
                ], style={"display": "flex", "alignItems": "center"}),
            ),

            dcc.Loading(
                html.Div(id="loading-trigger", style={"height": "2px"}),
                type="circle", color=C["accent"],
            ),

            html.Div(id="response-panel"),

        ], style={"width": "58%", "minWidth": "340px", "flexShrink": "0"}),

        # Right — charts
        html.Div([
            _section_card(
                _label("Win probability — top 10"),
                dcc.Graph(
                    id="prediction-chart",
                    figure=_empty_chart("Run a pre-race query to see predictions"),
                    config={"displayModeBar": False},
                    style={"margin": "-4px -8px"},
                ),
            ),

            # Actual results panel — shown for predictive queries only
            html.Div(id="results-panel"),

            _section_card(
                _label("Course elevation profile"),
                dcc.Graph(
                    id="course-chart",
                    figure=_empty_chart("Ask about a specific race to see the course"),
                    config={"displayModeBar": False},
                    style={"margin": "-4px -8px"},
                ),
            ),

        ], style={"flex": "1", "minWidth": "300px"}),

    ], style={
        "display": "flex", "gap": "20px",
        "padding": "24px 28px", "maxWidth": "1400px",
        "margin": "0 auto", "alignItems": "flex-start",
    }),

], style={"background": C["bg"], "minHeight": "100vh",
          "fontFamily": FONT, "color": C["text"]})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("query-input", "value"),
    [Input({"type": "suggestion-btn", "index": ALL}, "n_clicks")],
    prevent_initial_call=True,
)
def fill_from_suggestion(n_clicks_list):
    if not any(n_clicks_list):
        raise PreventUpdate
    triggered = ctx.triggered_id
    if triggered is None:
        raise PreventUpdate
    _, query = SUGGESTED[triggered["index"]]
    return query


@app.callback(
    Output("agent-result-store",    "data"),
    Output("prediction-text-store", "data"),
    Output("race-context-store",    "data"),
    Output("results-store",         "data"),
    Output("loading-trigger",       "children"),
    Input("run-btn", "n_clicks"),
    State("query-input", "value"),
    prevent_initial_call=True,
)
def run_query(n_clicks, query):
    if not query or not query.strip():
        raise PreventUpdate

    data = query_api(query.strip())

    store = {
        "response":   data.get("response", ""),
        "query_type": data.get("query_type", ""),
        "steps":      data.get("steps", []),
        "error":      data.get("error", ""),
        "elapsed_s":  data.get("elapsed_s", 0),
    }
    pred_text = data.get("prediction_text") or ""
    race_ctx  = data.get("race_context") or {}

    results_data = None
    if race_ctx and race_ctx.get("race_name") and race_ctx.get("year"):
        results_data = fetch_results(
            race_name=race_ctx["race_name"],
            year=race_ctx["year"],
            stage=race_ctx.get("stage"),
        )

    return store, pred_text, race_ctx, results_data, ""


@app.callback(
    Output("response-panel", "children"),
    Output("query-meta",     "children"),
    Input("agent-result-store", "data"),
    prevent_initial_call=True,
)
def update_response(data):
    if not data:
        raise PreventUpdate

    response   = data.get("response", "")
    query_type = data.get("query_type", "")
    steps      = data.get("steps", [])
    error      = data.get("error", "")
    elapsed    = data.get("elapsed_s", 0)

    TYPE_LABELS = {
        "STRUCTURED":      "Structured",
        "SEMANTIC_COURSE": "Course search",
        "SEMANTIC_RIDER":  "Rider search",
        "PREDICTIVE":      "Pre-race analysis",
        "HYBRID":          "Hybrid",
    }

    meta = [
        _badge(TYPE_LABELS.get(query_type, query_type) or "—", C["accent"]),
        _badge(" → ".join(steps) or "—"),
        _badge(f"{elapsed}s"),
    ]

    if error and not response:
        panel = _section_card(
            html.Div(f"⚠ {error}", style={"color": C["red"], "fontSize": "13px"}),
        )
        return panel, meta

    panel = _section_card(
        html.Div(_render_response(response), style={"lineHeight": "1.75", "fontSize": "14px"}),
    )
    return panel, meta


@app.callback(
    Output("prediction-chart", "figure"),
    Input("prediction-text-store", "data"),
    prevent_initial_call=True,
)
def update_prediction_chart(pred_text):
    if not pred_text:
        raise PreventUpdate
    return build_prediction_chart(pred_text)


@app.callback(
    Output("course-chart", "figure"),
    Input("race-context-store", "data"),
    prevent_initial_call=True,
)
def update_course_chart(race_ctx):
    if not race_ctx or not race_ctx.get("race_name"):
        raise PreventUpdate
    return build_course_chart(
        race_ctx["race_name"],
        race_ctx.get("year") or 2023,
        race_ctx.get("stage"),
    )


@app.callback(
    Output("results-panel", "children"),
    Input("results-store", "data"),
    prevent_initial_call=True,
)
def update_results_panel(results_data):
    """Render the actual results table when race context is available."""
    if not results_data or not results_data.get("found"):
        return None

    results  = results_data.get("results", [])
    if not results:
        return None

    race_name = results_data.get("race_name", "")
    year      = results_data.get("year", "")
    stage     = results_data.get("stage")
    label     = f"{year} {race_name}" + (f" Stage {stage}" if stage else "")

    rows = []
    for r in results:
        rank   = r["rank"]
        is_win = rank == 1
        if rank == 1:   badge_color = C["accent"]
        elif rank <= 3: badge_color = "#9CA3AF"
        else:           badge_color = C["border"]

        rows.append(html.Tr([
            html.Td(
                html.Span(str(rank), style={
                    "background": badge_color,
                    "color":      C["bg"] if rank <= 3 else C["secondary"],
                    "fontFamily": MONO, "fontSize": "11px", "fontWeight": "700",
                    "padding": "2px 6px", "borderRadius": "3px",
                    "display": "inline-block", "minWidth": "24px", "textAlign": "center",
                }),
                style={"padding": "5px 8px 5px 0", "width": "36px"},
            ),
            html.Td(r["rider"], style={
                "color":      C["text"] if is_win else C["secondary"],
                "fontWeight": "600" if is_win else "400",
                "fontSize":   "12px", "padding": "5px 8px 5px 0",
            }),
            html.Td(r.get("team", ""), style={
                "color": C["secondary"], "fontSize": "11px", "padding": "5px 0",
            }),
        ], style={
            "borderBottom": f"1px solid {C['border']}",
            "background":   "rgba(245,197,24,0.05)" if is_win else "transparent",
        }))

    return _section_card(
        _label(f"Actual results — {label}"),
        html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("#",     style={"color": C["secondary"], "fontFamily": MONO, "fontSize": "10px", "padding": "0 8px 8px 0", "textAlign": "left"}),
                    html.Th("Rider", style={"color": C["secondary"], "fontFamily": MONO, "fontSize": "10px", "padding": "0 8px 8px 0", "textAlign": "left"}),
                    html.Th("Team",  style={"color": C["secondary"], "fontFamily": MONO, "fontSize": "10px", "padding": "0 0 8px 0",   "textAlign": "left"}),
                ])),
                html.Tbody(rows),
            ],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
        html.Div(
            "Compare ML predictions above with actual finishers",
            style={
                "color": C["secondary"], "fontSize": "10px",
                "fontFamily": MONO, "marginTop": "12px", "letterSpacing": "0.06em",
            },
        ),
    )


@app.callback(
    Output("status-indicator", "style"),
    Input("health-interval", "n_intervals"),
    prevent_initial_call=False,
)
def update_health(n):
    ok = check_api_health()
    return {
        "width": "8px", "height": "8px",
        "borderRadius": "50%", "display": "inline-block",
        "background": C["green"] if ok else C["red"],
    }


# ---------------------------------------------------------------------------
# Response renderer
# ---------------------------------------------------------------------------

def _render_response(text: str) -> list:
    if not text:
        return [html.P("No response.", style={"color": C["secondary"]})]

    elements = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        elif stripped.startswith("# "):
            elements.append(html.H2(stripped[2:], style={
                "fontFamily": "'Barlow Condensed', sans-serif",
                "fontWeight": "700", "fontSize": "20px",
                "textTransform": "uppercase", "letterSpacing": "0.04em",
                "color": C["text"], "margin": "0 0 16px 0",
                "paddingBottom": "12px", "borderBottom": f"2px solid {C['accent']}",
            }))
        elif stripped.startswith("## "):
            elements.append(html.H3(stripped[3:], style={
                "fontFamily": "'Barlow Condensed', sans-serif",
                "fontWeight": "700", "fontSize": "15px",
                "textTransform": "uppercase", "letterSpacing": "0.06em",
                "color": C["accent"], "margin": "20px 0 8px 0",
            }))
        elif stripped.startswith("### "):
            elements.append(html.H4(stripped[4:], style={
                "fontWeight": "600", "fontSize": "13px",
                "color": C["text"], "margin": "14px 0 4px 0",
            }))
        elif stripped.startswith(("- ", "• ")):
            elements.append(html.Li(
                _inline_md(stripped[2:]),
                style={"color": C["secondary"], "fontSize": "13px", "marginBottom": "4px"},
            ))
        else:
            elements.append(html.P(
                _inline_md(stripped),
                style={"color": C["secondary"], "fontSize": "13px", "margin": "0 0 8px 0"},
            ))
    return elements


def _inline_md(text: str) -> list:
    parts  = re.split(r"\*\*(.+?)\*\*", text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            result.append(html.Strong(part, style={"color": C["text"], "fontWeight": "600"}))
        else:
            result.append(part)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Expose Flask server for gunicorn
server = app.server

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=False, port=8050)