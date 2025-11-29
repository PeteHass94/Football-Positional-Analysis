import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Pitch
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from utils.page_components import add_common_page_elements

add_common_page_elements()

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

BASE_MATCH_DIR = Path("data/matches")

MATCH_IDS = [
    1886347, 1899585, 1925299, 1953632, 1996435,
    2006229, 2011166, 2013725, 2015213, 2017461,
]

MAX_MINUTE = 100  # 0–99


# ---------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------

@st.cache_data
def load_match_metadata(match_id: int) -> dict:
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_match.json"
    with path.open() as f:
        return json.load(f)


@st.cache_data
def load_tracking_frames(match_id: int) -> list[dict]:
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"
    frames = []
    with path.open() as f:
        for line in f:
            frames.append(json.loads(line))
    return frames


@st.cache_data
def load_dynamic_events(match_id: int) -> pd.DataFrame:
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_dynamic_events.csv"
    return pd.read_csv(path)


@st.cache_data
def build_player_lookup(match_id: int) -> pd.DataFrame:
    meta = load_match_metadata(match_id)
    players = meta.get("players", [])
    if not players:
        return pd.DataFrame()

    df = pd.json_normalize(players, sep="_")
    team_map = {
        meta["home_team"]["id"]: meta["home_team"]["short_name"],
        meta["away_team"]["id"]: meta["away_team"]["short_name"],
    }
    df["team_name"] = df["team_id"].map(team_map)

    return df[
        [
            "id",
            "team_id",
            "team_name",
            "number",
            "short_name",
            "player_role_position_group",
            "player_role_name",
            "player_role_acronym",
        ]
    ]


# ---------------------------------------------------------
# Time helpers
# ---------------------------------------------------------

def timestamp_to_minute(ts: str) -> int:
    """
    Convert timestamp 'HH:MM:SS.xx' -> integer minute since start.
    Example: '01:23:45.0' -> 83
    """
    if not isinstance(ts, str):
        return 0
    parts = ts.split(":")
    if len(parts) != 3:
        return 0
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        return hours * 60 + minutes
    except Exception:
        return 0


def compute_event_minute_col(dyn: pd.DataFrame) -> pd.Series:
    """
    Extract minute from a 'MM:SS.f' style string in time_start.
    Example: '36:12.5' -> 36
    """
    if "time_start" in dyn.columns:
        return dyn["time_start"].astype(str).str.split(":", n=1, expand=True)[0].astype(int)
    elif "second_start" in dyn.columns:
        return (dyn["second_start"].fillna(0) // 60).astype(int)
    elif "minute_start" in dyn.columns:
        return dyn["minute_start"].fillna(0).astype(int)
    else:
        return pd.Series(0, index=dyn.index, dtype=int)


def is_shot_event(row: pd.Series) -> bool:
    """
    Treat an event as a shot only if end_type == 'shot' (case-insensitive).
    """
    if "end_type" not in row or pd.isna(row["end_type"]):
        return False
    return str(row["end_type"]).strip().lower() == "shot"

def is_goal_event(row: pd.Series) -> bool:
    """
    Treat an event as a goal if end_type == 'shot' and lead_to_goal is True (case-insensitive).
    """
    if "end_type" not in row or pd.isna(row["end_type"]):
        return False
    return str(row["end_type"]).strip().lower() == "shot" and row.get("lead_to_goal", False)

# ---------------------------------------------------------
# Aggregation of minute positions
# ---------------------------------------------------------

@st.cache_data
def aggregate_minute_positions(match_id: int):
    """
    Aggregate average positions per minute for home/away players and ball.

    Returns:
      minute_data: dict[minute] -> {
          "home": [{"x", "y", "player_id", "short_name", "number", "team_id"}...],
          "away": [...],
          "ball": {"x", "y", "n"}
      }
      minutes_sorted: sorted list of minutes that have frames (<= MAX_MINUTE-1)
    """
    meta = load_match_metadata(match_id)
    frames = load_tracking_frames(match_id)
    player_lookup = build_player_lookup(match_id)

    player_map = {row["id"]: row for _, row in player_lookup.iterrows()}

    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]

    minute_player_acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    minute_ball_acc = defaultdict(lambda: [0.0, 0.0, 0])

    for fr in frames:
        ts = fr.get("timestamp")
        minute = timestamp_to_minute(ts)
        if minute < 0 or minute >= MAX_MINUTE:
            continue

        # Ball
        ball = fr.get("ball_data") or {}
        if ball.get("is_detected") and ball.get("x") is not None and ball.get("y") is not None:
            b_acc = minute_ball_acc[minute]
            b_acc[0] += ball["x"]
            b_acc[1] += ball["y"]
            b_acc[2] += 1

        # Players
        for p in fr.get("player_data", []):
            pid, x, y = p.get("player_id"), p.get("x"), p.get("y")
            if pid is None or x is None or y is None:
                continue
            if pid not in player_map:
                continue
            acc = minute_player_acc[minute][pid]
            acc[0] += x
            acc[1] += y
            acc[2] += 1

    minute_data = {}
    for minute, players_dict in minute_player_acc.items():
        home_points, away_points = [], []

        for pid, (sx, sy, c) in players_dict.items():
            if c == 0:
                continue
            info = player_map.get(pid)
            if info is None:
                continue
            avg_x = sx / c
            avg_y = sy / c
            entry = {
                "player_id": pid,
                "x": avg_x,
                "y": avg_y,
                "short_name": info["short_name"],
                "number": info["number"],
                "team_id": info["team_id"],
            }
            if info["team_id"] == home_id:
                home_points.append(entry)
            elif info["team_id"] == away_id:
                away_points.append(entry)

        b_sx, b_sy, b_c = minute_ball_acc[minute]
        ball_info = None
        if b_c > 0:
            ball_info = {"x": b_sx / b_c, "y": b_sy / b_c, "n": b_c}

        minute_data[minute] = {
            "home": home_points,
            "away": away_points,
            "ball": ball_info,
        }

    minutes_sorted = sorted(minute_data.keys())
    return minute_data, minutes_sorted


# ---------------------------------------------------------
# Label: shot in next 5 minutes per team
# ---------------------------------------------------------

@st.cache_data
def compute_minute_targets(match_id: int) -> pd.DataFrame:
    """
    For each minute (0–MAX_MINUTE-1), compute per-team labels:
      - home_shot_next5
      - away_shot_next5
    based on dynamic events (end_type == 'shot') in window (m, m+5].
    """
    meta = load_match_metadata(match_id)
    dyn = load_dynamic_events(match_id).copy()

    dyn["minute"] = compute_event_minute_col(dyn)
    dyn["is_shot"] = dyn.apply(is_shot_event, axis=1)
    dyn["is_goal"] = dyn.apply(is_goal_event, axis=1)

    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]
    home_short = meta["home_team"]["short_name"]
    away_short = meta["away_team"]["short_name"]

    team_id_col = None
    team_short_col = None

    if "team_in_possession_id" in dyn.columns:
        team_id_col = "team_in_possession_id"
    elif "team_id" in dyn.columns:
        team_id_col = "team_id"

    if "team_in_possession_shortname" in dyn.columns:
        team_short_col = "team_in_possession_shortname"
    elif "team_in_possession_name" in dyn.columns:
        team_short_col = "team_in_possession_name"

    rows = []
    for minute in range(0, MAX_MINUTE):
        m_start = minute
        m_end = minute + 5

        mask_window = (dyn["minute"] > m_start) & (dyn["minute"] <= m_end)
        win = dyn[mask_window & dyn["is_shot"]]

        if win.empty:
            rows.append(
                {
                    "match_id": match_id,
                    "minute": minute,
                    "home_shot_next5": False,
                    "away_shot_next5": False,
                }
            )
            continue

        if team_id_col is not None:
            home_shots = win[win[team_id_col] == home_id]
            away_shots = win[win[team_id_col] == away_id]
        elif team_short_col is not None:
            home_shots = win[win[team_short_col] == home_short]
            away_shots = win[win[team_short_col] == away_short]
        else:
            # Cannot identify teams -> count all shots but can't split
            home_shots = pd.DataFrame(columns=win.columns)
            away_shots = pd.DataFrame(columns=win.columns)

        rows.append(
            {
                "match_id": match_id,
                "minute": minute,
                "home_shot_next5": not home_shots.empty,
                "away_shot_next5": not away_shots.empty,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Positional shape features (per team, per minute)
# ---------------------------------------------------------

def safe_convex_hull_area(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 3:
        return 0.0
    pts = np.column_stack([xs, ys])
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)  # in 2D, volume is area
    except Exception:
        return 0.0


def compute_team_shape_features(
    team_points: list[dict],
    ball_info: dict | None,
    team_role: str,
    pitch_length: float,
) -> dict:
    """
    team_role: 'home' or 'away'
    We assume x-axis is length (goals on left/right).
      - Home attacks in +x direction.
      - Away attacks in -x direction.
    """
    if not team_points:
        # no players tracked (shouldn't really happen)
        return {
            "n_players": 0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "spread_x": 0.0,
            "spread_y": 0.0,
            "hull_area": 0.0,
            "n_final_third": 0,
            "n_players_behind_ball": 0,
            "avg_dist_behind_ball": 0.0,
            "max_dist_behind_ball": 0.0,
            "ball_x": 0.0,
            "ball_y": 0.0,
            "dist_centroid_ball": 0.0,
        }

    xs = np.array([p["x"] for p in team_points], dtype=float)
    ys = np.array([p["y"] for p in team_points], dtype=float)

    n_players = len(xs)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    spread_x = float(xs.max() - xs.min())
    spread_y = float(ys.max() - ys.min())
    hull_area = safe_convex_hull_area(xs, ys)

    # Final third (very simple heuristic: last 1/3 of pitch in attacking direction)
    # Pitch goes roughly [-L/2, +L/2] on x.
    L = pitch_length
    third = L / 6.0  # boundary from centre to final third

    if team_role == "home":
        final_third_mask = xs > third
    else:  # away attacking negative x
        final_third_mask = xs < -third
    n_final_third = int(final_third_mask.sum())

    # Ball-based features
    if ball_info is not None:
        ball_x = float(ball_info["x"])
        ball_y = float(ball_info["y"])
    else:
        ball_x = 0.0
        ball_y = 0.0

    # Players behind ball (again relative to attacking direction)
    if ball_info is not None:
        if team_role == "home":
            behind_mask = xs < ball_x
            dists = ball_x - xs[behind_mask]
        else:
            behind_mask = xs > ball_x
            dists = xs[behind_mask] - ball_x

        n_players_behind_ball = int(behind_mask.sum())
        if len(dists) > 0:
            avg_dist_behind_ball = float(dists.mean())
            max_dist_behind_ball = float(dists.max())
        else:
            avg_dist_behind_ball = 0.0
            max_dist_behind_ball = 0.0
    else:
        n_players_behind_ball = 0
        avg_dist_behind_ball = 0.0
        max_dist_behind_ball = 0.0

    dist_centroid_ball = float(
        np.sqrt((centroid_x - ball_x) ** 2 + (centroid_y - ball_y) ** 2)
    )

    return {
        "n_players": n_players,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "spread_x": spread_x,
        "spread_y": spread_y,
        "hull_area": hull_area,
        "n_final_third": n_final_third,
        "n_players_behind_ball": n_players_behind_ball,
        "avg_dist_behind_ball": avg_dist_behind_ball,
        "max_dist_behind_ball": max_dist_behind_ball,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "dist_centroid_ball": dist_centroid_ball,
    }


@st.cache_data
def build_match_minute_dataset(match_id: int) -> pd.DataFrame:
    """
    Build a dataframe of per-minute, per-team features + labels for one match.

    Columns include:
      - match_id, minute, team_role ('home'/'away'), team_short
      - shape features
      - label_shot_next5 (0/1)
    """
    meta = load_match_metadata(match_id)
    minute_data, minutes_sorted = aggregate_minute_positions(match_id)
    minute_targets = compute_minute_targets(match_id)

    home_short = meta["home_team"]["short_name"]
    away_short = meta["away_team"]["short_name"]

    rows = []

    for minute in minutes_sorted:
        minute_entry = minute_data.get(minute)
        if minute_entry is None:
            continue

        ball_info = minute_entry["ball"]
        pitch_length = meta["pitch_length"]

        target_row = minute_targets[minute_targets["minute"] == minute]
        if target_row.empty:
            home_shot_label = False
            away_shot_label = False
        else:
            home_shot_label = bool(target_row.iloc[0]["home_shot_next5"])
            away_shot_label = bool(target_row.iloc[0]["away_shot_next5"])

        # Home team
        home_points = minute_entry["home"]
        if home_points:
            feats_home = compute_team_shape_features(
                home_points, ball_info, "home", pitch_length
            )
            feats_home.update(
                {
                    "match_id": match_id,
                    "minute": minute,
                    "team_role": "home",
                    "team_short": home_short,
                    "label_shot_next5": int(home_shot_label),
                }
            )
            rows.append(feats_home)

        # Away team
        away_points = minute_entry["away"]
        if away_points:
            feats_away = compute_team_shape_features(
                away_points, ball_info, "away", pitch_length
            )
            feats_away.update(
                {
                    "match_id": match_id,
                    "minute": minute,
                    "team_role": "away",
                    "team_short": away_short,
                    "label_shot_next5": int(away_shot_label),
                }
            )
            rows.append(feats_away)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


@st.cache_data
def build_full_dataset() -> pd.DataFrame:
    dfs = []
    for mid in MATCH_IDS:
        df_match = build_match_minute_dataset(mid)
        if not df_match.empty:
            dfs.append(df_match)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def plot_shape_for_minute(meta: dict, minute_entry: dict, team_role: str):
    """
    Visualise the team shape + ball for a single minute & team.
    Highlights the chosen team and shows other team more faintly.
    """
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=meta["pitch_length"],
        pitch_width=meta["pitch_width"],
        line_zorder=1,
    )
    fig, ax = pitch.draw(figsize=(5, 4))

    home_pts = minute_entry["home"]
    away_pts = minute_entry["away"]
    ball = minute_entry["ball"]

    if team_role == "home":
        primary_pts = home_pts
        secondary_pts = away_pts
        primary_kit = meta["home_team_kit"]
        secondary_kit = meta["away_team_kit"]
        primary_name = meta["home_team"]["short_name"]
        secondary_name = meta["away_team"]["short_name"]
    else:
        primary_pts = away_pts
        secondary_pts = home_pts
        primary_kit = meta["away_team_kit"]
        secondary_kit = meta["home_team_kit"]
        primary_name = meta["away_team"]["short_name"]
        secondary_name = meta["home_team"]["short_name"]

    # Primary team (highlight)
    if primary_pts:
        xs = np.array([p["x"] for p in primary_pts])
        ys = np.array([p["y"] for p in primary_pts])
        pitch.scatter(
            xs,
            ys,
            ax=ax,
            facecolor=primary_kit["jersey_color"],
            edgecolor=primary_kit["number_color"],
            s=80,
            label=primary_name,
        )

        # Convex hull (compactness/spread)
        if len(xs) >= 3:
            try:
                hull = ConvexHull(np.column_stack([xs, ys]))
                hull_points = np.column_stack([xs, ys])[hull.vertices]
                ax.fill(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    alpha=0.15,
                    color= "grey" if primary_kit["jersey_color"] == '#ffffff' else primary_kit["jersey_color"],
                    zorder=1,
                )
            except Exception:
                pass

        # Centroid
        cx = xs.mean()
        cy = ys.mean()
        ax.scatter(
            cx,
            cy,
            marker="X",
            s=120,
            color="black",
            zorder=4,
            label="Team centroid",
        )

    # Secondary team (faded)
    if secondary_pts:
        xs2 = [p["x"] for p in secondary_pts]
        ys2 = [p["y"] for p in secondary_pts]
        pitch.scatter(
            xs2,
            ys2,
            ax=ax,
            facecolor=secondary_kit["jersey_color"],
            edgecolor=secondary_kit["number_color"],
            s=60,
            alpha=0.25,
            label=secondary_name,
        )

    # Ball
    if ball is not None:
        pitch.scatter(
            ball["x"],
            ball["y"],
            ax=ax,
            marker="football",
            facecolor="white",
            edgecolors="black",
            s=140,
            zorder=4,
            label="Ball",
        )

    ax.set_title(f"Team shape for {primary_name}", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=8)
    return fig


# ---------------------------------------------------------
# Streamlit Page
# ---------------------------------------------------------

def main():
    st.title("📈 Shot Creation Probability From Positional Shape")

    st.markdown(
        """
        This page builds a **per-minute, per-team dataset** from tracking + dynamic events.

        For each (match, minute, team) we compute:
        - Convex hull area (team compactness/spread)
        - Team centroid (x, y)
        - Horizontal/vertical spread
        - Number of players in the attacking final third
        - How many players are **behind the ball** and how far
        - Ball position and distance from team centroid

        Then we label each row as **1** if that team takes a shot in the next 5 minutes
        (based on dynamic events `end_type == 'shot'`), otherwise **0**.

        We train:
        - A **Logistic Regression** model (interpretable)
        - An **XGBoost** model (more flexible)

        And we show:
        - A **timeline** of shot probability over the match
        - A **pitch heatmap** of risky ball positions
        - The **most important features** driving shot creation
        """
    )

    # -----------------------------------------------------
    # Build dataset
    # -----------------------------------------------------
    st.header("1. Build dataset")

    with st.spinner("Building per-minute positional dataset across all matches..."):
        df_all = build_full_dataset()

    if df_all.empty:
        st.error("No data available. Check that matches and files are in data/matches.")
        return

    st.write(f"Total samples (minutes × teams): **{len(df_all)}**")
    st.write("Preview of the engineered dataset:")
    st.dataframe(df_all)

    # -----------------------------------------------------
    # 2. Inspect feature working for a single match / minute
    # -----------------------------------------------------
    st.header("2. Inspect features minute-by-minute")

    match_id_vis = st.selectbox(
        "Match to inspect",
        MATCH_IDS,
        index=0,
        key="inspect_match",
    )
    meta_vis = load_match_metadata(match_id_vis)
    # -----------------------------------------------------
    # Match overview
    # -----------------------------------------------------
    st.header("Match overview")

    home = meta_vis["home_team"]["short_name"]
    away = meta_vis["away_team"]["short_name"]
    score = f'{meta_vis["home_team_score"]}–{meta_vis["away_team_score"]}'
    kick_off = meta_vis["date_time"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Home team", home)
    with col2:
        st.metric("Away team", away)
    with col3:
        st.metric("Final score", score)

    col4, col5 = st.columns(2)
    with col4:
        st.write(
            f"**Competition**: {meta_vis['competition_edition']['competition']['name']} "
            f"({meta_vis['competition_edition']['season']['name']})"
        )
        st.write(f"**Round**: {meta_vis['competition_round']['name']}")
        st.write(f"**Kick-off (UTC)**: {kick_off}")
    with col5:
        stadium = meta_vis["stadium"]
        st.write(f"**Stadium**: {stadium['name']} — {stadium['city']}")
        st.write(f"**Capacity**: {stadium['capacity']}")
        st.write(f"**Pitch size**: {meta_vis['pitch_length']}m × {meta_vis['pitch_width']}m")
        
    with st.expander("Show full phase JSON"):
        st.json(meta_vis, expanded=False)

    st.markdown("---")
    
    minute_data_vis, minutes_sorted_vis = aggregate_minute_positions(match_id_vis)

    home_short_vis = meta_vis["home_team"]["short_name"]
    away_short_vis = meta_vis["away_team"]["short_name"]

    team_short_vis = st.radio(
        "Team",
        options=[home_short_vis, away_short_vis],
        horizontal=True,
        key="inspect_team",
    )

    # rows for this match + team
    df_match_vis = df_all[
        (df_all["match_id"] == match_id_vis)
        & (df_all["team_short"] == team_short_vis)
    ].copy()

    if df_match_vis.empty:
        st.warning("No samples for this match/team.")
    else:
        minutes_avail = sorted(df_match_vis["minute"].unique())
        min_minute = int(min(minutes_avail))
        max_minute = int(max(minutes_avail))

        minute_sel = st.slider(
            "Minute",
            min_value=min_minute,
            max_value=max_minute,
            value=min_minute,
            step=1,
            key="inspect_minute",
        )

        row = df_match_vis[df_match_vis["minute"] == minute_sel]
        if row.empty:
            st.warning("No data for this minute.")
        else:
            row = row.iloc[0]  # single row
            team_role = row["team_role"]  # 'home' or 'away'

            st.subheader(f"Features for minute {minute_sel}")

            # Show key features + label in a tidy table
            feature_cols = [
                "n_players",
                "centroid_x",
                "centroid_y",
                "spread_x",
                "spread_y",
                "hull_area",
                "n_final_third",
                "n_players_behind_ball",
                "avg_dist_behind_ball",
                "max_dist_behind_ball",
                "ball_x",
                "ball_y",
                "dist_centroid_ball",
            ]

            display_cols = (
                ["match_id", "team_short", "team_role", "minute", "label_shot_next5"]
                + feature_cols
            )
            st.table(row[display_cols].to_frame().T)

            # And the corresponding pitch visualisation
            st.subheader("Positional shape for this minute")

            minute_entry = minute_data_vis.get(minute_sel)
            if minute_entry is None:
                st.warning("No tracking data for this minute.")
            else:
                fig_shape = plot_shape_for_minute(meta_vis, minute_entry, team_role)
                st.pyplot(fig_shape, use_container_width=True)

                st.caption(
                    "Convex hull shows the team’s spread; the X marker is the centroid; "
                    "ball icon is the average ball position for this minute."
                )

    st.markdown("---")


    # -----------------------------------------------------
    # Prepare data for model training
    # -----------------------------------------------------
    
    
    # Feature columns
    feature_cols = [
        "n_players",
        "centroid_x",
        "centroid_y",
        "spread_x",
        "spread_y",
        "hull_area",
        "n_final_third",
        "n_players_behind_ball",
        "avg_dist_behind_ball",
        "max_dist_behind_ball",
        "ball_x",
        "ball_y",
        "dist_centroid_ball",
    ]

    X = df_all[feature_cols].values
    y = df_all["label_shot_next5"].values

    # -----------------------------------------------------
    # Train models
    # -----------------------------------------------------
    st.header("3. Train models")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError:
        st.error("Please install scikit-learn: `pip install scikit-learn`")
        return

    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        xgb_available = False
        st.warning("XGBoost not installed. Install with `pip install xgboost` to use it.")

    test_size = st.slider("Test size fraction", 0.1, 0.4, 0.25, 0.05)
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Logistic Regression (with standardization)
    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_pred_lr)

    st.write(f"**Logistic Regression AUC:** {auc_lr:.3f}")

    # XGBoost
    if xgb_available:
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_pred_xgb)
        st.write(f"**XGBoost AUC:** {auc_xgb:.3f}")
    else:
        xgb_model = None

    st.markdown("---")

    # -----------------------------------------------------
    # Timeline chart for a chosen match/team
    # -----------------------------------------------------
    st.header("4. Timeline: shot probability over a match")

    match_id_sel = st.selectbox(
        "Select match for visualisation", MATCH_IDS, index=0, key="timeline_match"
    )
    meta_sel = load_match_metadata(match_id_sel)
    home_short = meta_sel["home_team"]["short_name"]
    away_short = meta_sel["away_team"]["short_name"]

    team_short_sel = st.radio(
        "Select team",
        options=[home_short, away_short],
        horizontal=True,
    )

    # Filter for this match + team
    df_match = df_all[
        (df_all["match_id"] == match_id_sel)
        & (df_all["team_short"] == team_short_sel)
    ].copy()

    if df_match.empty:
        st.warning("No samples for this match/team.")
        return

    # Get probabilities
    X_match = df_match[feature_cols].values
    df_match["proba_lr"] = lr_pipe.predict_proba(X_match)[:, 1]
    if xgb_model is not None:
        df_match["proba_xgb"] = xgb_model.predict_proba(X_match)[:, 1]

    df_match = df_match.sort_values("minute")

    model_for_timeline = st.radio(
        "Model for timeline",
        options=["Logistic Regression", "XGBoost"] if xgb_model is not None else ["Logistic Regression"],
        horizontal=True,
    )
    proba_col = "proba_lr" if model_for_timeline == "Logistic Regression" else "proba_xgb"

    # -----------------------------
    # Load dynamic events to get shots/goals
    # -----------------------------
    dyn = load_dynamic_events(match_id_sel).copy()
    dyn["minute"] = compute_event_minute_col(dyn)
    dyn["is_shot"] = dyn.apply(is_shot_event, axis=1)
    dyn["is_goal"] = dyn.apply(is_goal_event, axis=1)
    
    # st.dataframe(dyn[['minute', 'is_shot', 'is_goal']])
    
    # Filter to events for the selected team
    team_id = meta_sel["home_team"]["id"] if team_short_sel == home_short else meta_sel["away_team"]["id"]

    # Best guess for team columns
    if "team_in_possession_id" in dyn.columns:
        team_mask = dyn["team_in_possession_id"] == team_id
    elif "team_id" in dyn.columns:
        team_mask = dyn["team_id"] == team_id
    elif "team_in_possession_shortname" in dyn.columns:
        team_mask = dyn["team_in_possession_shortname"] == team_short_sel
    else:
        team_mask = pd.Series(False, index=dyn.index)

    dyn_team = dyn[team_mask & dyn["is_shot"]].copy()

    # Separate shots and goals
    shot_minutes = dyn_team["minute"].tolist()
    goal_minutes = dyn_team[dyn_team["is_goal"]]["minute"].tolist()

    # -----------------------------
    # Build Plotly Figure
    # -----------------------------
    import plotly.graph_objects as go

    fig = go.Figure()

    # Probability line
    fig.add_trace(
        go.Scatter(
            x=df_match["minute"],
            y=df_match[proba_col],
            mode="lines",
            line=dict(width=3, color="#1f77b4"),
            name=f"{team_short_sel} probability",
        )
    )

    # Vertical shot markers (red)
    for m in shot_minutes:
        fig.add_trace(
            go.Scatter(
                x=[m, m],
                y=[0, 1],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                hovertemplate=f"Shot at minute {m}<extra></extra>",
                name="Shot",
                showlegend=False,
            )
        )

    # Vertical goal markers (green, on top)
    for m in goal_minutes:
        fig.add_trace(
            go.Scatter(
                x=[m, m],
                y=[0, 1],
                mode="lines",
                line=dict(color="green", width=3),
                hovertemplate=f"GOAL at minute {m}<extra></extra>",
                name="Goal",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(range=[0, 1], title="Probability"),
        xaxis=dict(title="Minute"),
        title=f"Shot Creation Probability — {team_short_sel}",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Vertical lines show **shots (red)** and **shots that lead to goals (green)** for {team_short_sel}. "
        f"Curve shows predicted probability of creating a shot in the next 5 minutes."
    )

    st.markdown("---")


    # -----------------------------------------------------
    # Pitch heatmap of risky ball positions
    # -----------------------------------------------------
    st.header("5. Pitch heatmap of risky positions")

    st.markdown(
        """
        ### 🔍 What this heatmap shows
        
        For every minute in the dataset, we know:
        - where the **ball** was (x, y on the pitch), and  
        - the model’s predicted probability that the team will create a **shot within the next 5 minutes**.
        
        We group the entire pitch into small spatial bins and compute:

        **mean shot probability in each pitch zone, given the ball is in that zone.**

        This gives a clear, data-driven picture of **where on the pitch possession tends to become dangerous** for this team.

        #### Why this matters
        - If certain wide areas consistently show higher probability → team creates danger when attacking down the wings.  
        - If central deep zones are high → team builds through the middle before accelerating.  
        - If final-third zones explode in probability → team is a classic “territory = threat” side.
        - If most dangerous zones are in their own half → team is a fast-transition / counter team.

        This is essentially a *positional expected threat* model derived from your ML shot-creation model.
        """
    )

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=meta_sel["pitch_length"],
        pitch_width=meta_sel["pitch_width"],
        line_zorder=1,
    )

    # Filter all minutes for the selected team
    df_heat = df_all[df_all["team_short"] == team_short_sel].copy()
    X_heat = df_heat[feature_cols].values
    df_heat["proba_lr"] = lr_pipe.predict_proba(X_heat)[:, 1]
    if xgb_model is not None:
        df_heat["proba_xgb"] = xgb_model.predict_proba(X_heat)[:, 1]

    proba_heat_col = (
        "proba_lr" if model_for_timeline == "Logistic Regression" else "proba_xgb"
    )

    # Compute binned probability
    stat = pitch.bin_statistic(
        df_heat["ball_x"].values,
        df_heat["ball_y"].values,
        values=df_heat[proba_heat_col].values,
        statistic="mean",
        bins=(12, 8),
    )

    fig, ax = pitch.draw(figsize=(6, 4))
    pcm = pitch.heatmap(stat, ax=ax, cmap="viridis")
    pitch.scatter(
        df_heat["ball_x"].values,
        df_heat["ball_y"].values,
        ax=ax,
        s=8,
        alpha=0.25,
        edgecolors="black",
    )
    ax.set_title(
        f"Average {model_for_timeline} shot probability\n"
        f"by ball position for {team_short_sel}",
        fontsize=12,
    )
    plt.colorbar(
        pcm, ax=ax, fraction=0.03, pad=0.02, label="Shot probability (next 5 mins)"
    )
    st.pyplot(fig, use_container_width=True)

    # --------------------------------------------------
    # Evidence summary (wide vs central, deep vs high)
    # --------------------------------------------------

    st.subheader("Summary: areas with increased")

    df_h = df_heat.copy()
    median_pitch_x = meta_sel["pitch_length"] / 2
    wide_threshold = meta_sel["pitch_width"] * 0.25  # 25% each side

    # Categorise zones
    df_h["side"] = pd.cut(
        df_h["ball_y"],
        bins=[-999, -wide_threshold, wide_threshold, 999],
        labels=["left wide (25%)", "central (50%)", "right wide (25%)"],
    )

    df_h["depth"] = pd.cut(
        df_h["ball_x"],
        bins=[-999, median_pitch_x * 0.33, median_pitch_x * 0.66, 999],
        labels=["own 3rd", "middle 3rd", "final 3rd"],
    )

    wide_summary = df_h.groupby("side")[proba_heat_col].mean().round(3)
    depth_summary = df_h.groupby("depth")[proba_heat_col].mean().round(3)

    c1, c2 = st.columns(2)

    with c1:
        st.write("### 📍 By width (left/middle/right), Avg. Prob.")
        st.dataframe(wide_summary)

    with c2:
        st.write("### 📍 By pitch depth (own/middle/final 3rd), Avg. Prob.")
        st.dataframe(depth_summary)

    # st.markdown(
    #     """
    #     **Interpretation:**

    #     These tables give direct evidence to support statements like:
    #     - *“This team generates more danger when the ball is wide”*  
    #     - *“Probability spikes only once the ball enters the final third”*  
    #     - *“Danger emerges earlier in build-up (middle 3rd) rather than final 3rd”*  
    #     - *“Central zones are safer/dangerous compared to wide zones”*  
        
    #     You now have both:
    #     - the **heatmap** (spatial pattern), and  
    #     - the **summary statistics** (quantitative evidence)  
    #     to make tactical insights that are grounded in the data.
    #     """
    # )

    st.markdown("---")


    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------
    st.header("6. Feature importance")

    st.markdown("""
    The model uses spatial + structural features of team shape to estimate how likely
    a team is to create a **shot in the next 5 minutes**.  
    Below is a reference explaining what each feature means and why it matters.
    """)

    # ---------------------------
    # Feature explanation table
    # ---------------------------

    feature_explanations = {
        "spread_y": "Vertical spread — how tall the team shape is (distance between deepest & highest player). Larger spread often means stretching the pitch vertically to attack space.",
        "dist_centroid_ball": "Distance between ball and team centroid — measures how 'connected' the team is to the ball. Large values indicate the ball is ahead of the block (e.g., counterattack or advanced progression).",
        "ball_y": "The ball's lateral (side-to-side) position — tells whether play is on the left, centre, or right wing.",
        "max_dist_behind_ball": "Maximum distance of any player behind the ball — captures extreme stagger (e.g., last defender far behind). Can relate to rest defence or transitions.",
        "n_players": "Number of detected players contributing to the minute average — larger numbers usually mean stable shape and good structure.",
        "spread_x": "Horizontal spread — team width across the pitch. Wide attacks often disrupt defensive blocks.",
        "hull_area": "Convex hull area of the team — total surface area of team shape. Big area = stretched, expansive structure; small area = compact block.",
        "avg_dist_behind_ball": "Average distance of players behind the ball — measures how deep the supporting structure is behind possession.",
        "centroid_y": "Y-position of the team centroid — how left/right the block is shifted.",
        "n_players_behind_ball": "How many players are behind the ball — relates to support structure and defensive balance.",
        "ball_x": "Ball's vertical field position — deeper or higher up the pitch. Strongest predictor for most build-up → shot models.",
        "n_final_third": "Number of players positioned in the attacking final third — more players high = more attacking intent.",
        "centroid_x": "X-position of the team centroid — average vertical height of the entire team. Captures how far the block has advanced.",
    }

    df_feat_exp = pd.DataFrame(
        [{"feature": k, "meaning": v} for k, v in feature_explanations.items()]
    )

    st.subheader("📘 Feature meanings (quick reference)")
    st.dataframe(df_feat_exp, use_container_width=True)

    st.markdown("""
    ---  
    ## Interpreting coefficient signs  
    **Logistic Regression coefficients represent how each feature influences the likelihood of a shot occurring in the next 5 minutes.**  
    - **Positive coefficient → increases shot probability**  
      (“More of this feature → more likely to create a shot soon”)  
    - **Negative coefficient → decreases shot probability**  
      (“More of this feature → less likely to create a shot soon”)  

    Common tactical interpretations:
    - Big **positive** on `ball_x` → more danger the further up the pitch the ball is.  
    - Positive on `spread_x` or `spread_y` → wide or vertically stretched structures create chances.  
    - Positive on `n_final_third` → advanced occupation predicts shot events.  
    - Negative on `avg_dist_behind_ball` → too many players sitting deep reduces attacking potency.  
    - Negative on `n_players_behind_ball` → fewer players behind ball often means faster attacks / transitions.  

    ---
    """)

    # ---------------------------
    # Logistic Regression Coeffs
    # ---------------------------
    lr_clf = lr_pipe.named_steps["clf"]
    coef = lr_clf.coef_[0]
    df_coef = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False)

    st.subheader("📈 Logistic Regression coefficients")
    st.dataframe(df_coef)

    # ---------------------------
    # XGBoost Importances
    # ---------------------------
    if xgb_model is not None:
        importances = xgb_model.feature_importances_
        df_imp = pd.DataFrame(
            {"feature": feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

        st.subheader("🌲 XGBoost feature importance")
        st.dataframe(df_imp)


if __name__ == "__main__":
    main()
