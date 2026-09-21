from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700&display=swap');
        :root {
            --ink: #08110f;
            --panel: #101c18;
            --panel-2: #152620;
            --line: #29443a;
            --signal: #b8f34a;
            --cyan: #4de2ce;
            --paper: #edf5e8;
            --muted: #9bb0a7;
        }
        .stApp {
            color: var(--paper);
            background:
                linear-gradient(rgba(77,226,206,.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(77,226,206,.035) 1px, transparent 1px),
                radial-gradient(circle at 82% 8%, rgba(184,243,74,.13), transparent 24rem),
                var(--ink);
            background-size: 34px 34px, 34px 34px, auto, auto;
            font-family: 'Manrope', sans-serif;
        }
        .stApp > header, .stAppHeader, header[data-testid="stHeader"] {
            background: rgba(8,17,15,.98) !important; border-bottom: 1px solid var(--line);
        }
        [data-testid="stToolbar"] { color: var(--paper); }
        h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -.035em; }
        code, .stCode, [data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.4rem 2.5rem 2.2rem;
            margin: .2rem 0 1.4rem;
            border: 1px solid var(--line);
            border-radius: 4px;
            background: linear-gradient(115deg, rgba(16,28,24,.97), rgba(21,38,32,.84));
            box-shadow: 0 24px 80px rgba(0,0,0,.28);
        }
        .hero:after {
            content: 'N≡N  →  NH₃';
            position: absolute;
            right: 2rem;
            top: 1.25rem;
            color: rgba(184,243,74,.10);
            font: 700 4.6rem 'DM Mono', monospace;
            transform: rotate(-4deg);
        }
        .eyebrow { color: var(--signal); font: 500 .74rem 'DM Mono', monospace; letter-spacing: .16em; }
        .hero h1 { margin: .45rem 0 .3rem; font-size: clamp(2.2rem, 5vw, 4.3rem); line-height: .95; }
        .hero p { max-width: 48rem; color: var(--muted); font-size: 1rem; }
        .status-strip { display: flex; gap: .65rem; flex-wrap: wrap; margin-top: 1.2rem; }
        .status-pill {
            border: 1px solid var(--line); background: rgba(8,17,15,.62);
            padding: .38rem .65rem; color: var(--cyan); font: 500 .72rem 'DM Mono', monospace;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .4rem; }
        [data-testid="stTabs"] button {
            border: 1px solid var(--line); border-radius: 2px; padding: .7rem 1rem;
            background: rgba(16,28,24,.8); color: var(--muted) !important; opacity: 1;
        }
        [data-testid="stTabs"] button[aria-selected="true"] { color: var(--signal); border-color: var(--signal); }
        [data-testid="stTabs"] button[aria-selected="true"] p { color: var(--signal) !important; }
        [data-testid="stTabs"] button p,
        [data-testid="stTabs"] button[aria-selected="false"] * {
            color: var(--muted) !important; opacity: 1 !important;
        }
        button[role="tab"], button[role="tab"] * {
            color: var(--muted) !important; opacity: 1 !important;
        }
        button[role="tab"][aria-selected="true"], button[role="tab"][aria-selected="true"] * {
            color: var(--signal) !important;
        }
        [data-baseweb="tab-highlight"] { background-color: var(--signal) !important; }
        [data-testid="stSidebar"] { background: #0c1714; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
        label[data-testid="stWidgetLabel"] p, [data-testid="stRadio"] label p,
        [data-baseweb="radio"] div {
            color: var(--muted) !important;
        }
        [data-testid="stSidebar"] strong { color: var(--paper); }
        label, label *, [data-testid="stAlert"] * { color: var(--muted) !important; opacity: 1 !important; }
        div[data-testid="stForm"], div[data-testid="stExpander"] {
            border-color: var(--line); background: rgba(16,28,24,.72); border-radius: 3px;
        }
        .stButton > button, .stFormSubmitButton > button {
            border-radius: 2px; border: 1px solid var(--signal); background: var(--signal);
            color: #10200e; font-weight: 700;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            border-color: var(--cyan); background: var(--cyan); color: #07110f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
