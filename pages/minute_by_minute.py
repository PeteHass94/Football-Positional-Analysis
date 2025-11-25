import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st
from mplsoccer import Pitch

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

MAX_MINUTE = 100  # upper limit for the grid (0–99)


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
# Helpers: time & aggregation
# ---------------------------------------------------------

def timestamp_to_minute(ts: str) -> int:
    """
    Convert a timestamp like '01:36:50.80' to a match minute (0–...).
    minute = hours * 60 + minutes (seconds ignored for binning).
    """
    if not isinstance(ts, str):
        return 0
    parts = ts.split(":")
    if len(parts) != 3:
        return 0
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        # seconds = float(parts[2])  # not needed for minute bin
        return hours * 60 + minutes
    except Exception:
        return 0


@st.cache_data
def aggregate_minute_positions(match_id: int):
    """
    Aggregate average positions per minute for home/away players and ball.
    Returns:
      minute_data: dict[minute] -> {
          "home": [{"x", "y", "player_id", "short_name", "number"}...],
          "away": [...],
          "ball": {"x", "y", "n"}
      }
      minutes_sorted: sorted list of minutes that actually have frames (<= MAX_MINUTE-1)
    """
    meta = load_match_metadata(match_id)
    frames = load_tracking_frames(match_id)
    player_lookup = build_player_lookup(match_id)

    # Build player lookup dict for speed
    player_map = {row["id"]: row for _, row in player_lookup.iterrows()}

    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]

    # Accumulators: minute -> player_id -> (sum_x, sum_y, count)
    minute_player_acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    # Ball accumulators: minute -> [sum_x, sum_y, count]
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

    # Build output structure
    minute_data = {}
    for minute, players_dict in minute_player_acc.items():
        home_points = []
        away_points = []

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

        # Ball
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


def compute_event_minute_col(dyn: pd.DataFrame) -> pd.Series:
    """
    Compute a 'minute' column for dynamic events using time_start, second_start or minute_start.
    """
    if "time_start" in dyn.columns:
        # assume HH:MM:SS or MM:SS – we only care about hour+minute
        return dyn["time_start"].astype(str).apply(timestamp_to_minute)
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


@st.cache_data
def compute_minute_targets(match_id: int) -> pd.DataFrame:
    """
    For each minute (0–MAX_MINUTE-1), compute:
      - shot_next5 (bool): True if ANY dynamic event with:
          minute > current_minute AND minute <= current_minute + 5
        has end_type == 'shot'.
    """
    dyn = load_dynamic_events(match_id).copy()

    # compute event minute
    dyn["minute"] = compute_event_minute_col(dyn)

    # identify shot events
    dyn["is_shot"] = dyn.apply(is_shot_event, axis=1)

    rows = []
    for minute in range(0, MAX_MINUTE):
        m_start = minute      # current minute
        m_end = minute + 5    # 5-minute horizon

        # event_minute > current_minute AND <= current_minute + 5
        mask_window = (dyn["minute"] > m_start) & (dyn["minute"] <= m_end)
        win = dyn[mask_window]

        shot_next5 = bool(win["is_shot"].any())

        rows.append(
            {
                "minute": minute,
                "shot_next5": shot_next5,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Plotting helper for a single minute
# ---------------------------------------------------------

def plot_minute_pitch(minute: int, meta: dict, minute_data: dict):
    """
    Plot average positions for a given minute using mplsoccer Pitch.
    """
    data = minute_data.get(minute)
    if data is None:
        return None

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=meta["pitch_length"],
        pitch_width=meta["pitch_width"],
        line_zorder=1,
    )
    fig, ax = pitch.draw(figsize=(4, 3))

    # Home players
    home_points = data["home"]
    if home_points:
        xs = [p["x"] for p in home_points]
        ys = [p["y"] for p in home_points]
        pitch.scatter(
            xs,
            ys,
            ax=ax,
            facecolor=meta["home_team_kit"]["jersey_color"],
            edgecolor=meta["home_team_kit"]["number_color"],
            s=60,
            label=meta["home_team"]["short_name"],
        )

    # Away players
    away_points = data["away"]
    if away_points:
        xs = [p["x"] for p in away_points]
        ys = [p["y"] for p in away_points]
        pitch.scatter(
            xs,
            ys,
            ax=ax,
            facecolor=meta["away_team_kit"]["jersey_color"],
            edgecolor=meta["away_team_kit"]["number_color"],
            s=60,
            label=meta["away_team"]["short_name"],
        )

    # Ball
    ball = data["ball"]
    if ball is not None:
        pitch.scatter(
            ball["x"],
            ball["y"],
            ax=ax,
            marker="football",
            facecolor="white",
            edgecolors="black",
            s=120,
            label="Ball",
        )

    ax.set_title(f"Minute {minute}", fontsize=10)
    return fig


# ---------------------------------------------------------
# Streamlit Page
# ---------------------------------------------------------

def main():
    st.title("🕒 Minute-by-minute summaries")

    st.markdown(
        """
        For the selected match, this page shows **average positions** of all players and the ball  
        for each match minute (0–99), and labels each minute with:
        
        - Whether the **home team** will have a shot in the next 5 minutes  
        - The **total xshot_player_possession_max** for each team in that 5-minute window  
        """
    )

    match_id = st.selectbox("Select match", MATCH_IDS, index=0)

    meta = load_match_metadata(match_id)
    
    # -----------------------------------------------------
    # Match overview
    # -----------------------------------------------------
    st.header("Match overview")

    home = meta["home_team"]["short_name"]
    away = meta["away_team"]["short_name"]
    score = f'{meta["home_team_score"]}–{meta["away_team_score"]}'
    kick_off = meta["date_time"]

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
            f"**Competition**: {meta['competition_edition']['competition']['name']} "
            f"({meta['competition_edition']['season']['name']})"
        )
        st.write(f"**Round**: {meta['competition_round']['name']}")
        st.write(f"**Kick-off (UTC)**: {kick_off}")
    with col5:
        stadium = meta["stadium"]
        st.write(f"**Stadium**: {stadium['name']} — {stadium['city']}")
        st.write(f"**Capacity**: {stadium['capacity']}")
        st.write(f"**Pitch size**: {meta['pitch_length']}m × {meta['pitch_width']}m")
        
    with st.expander("Show full phase JSON"):
        st.json(meta, expanded=False)

    st.markdown("---")
    
    minute_data, minutes_sorted = aggregate_minute_positions(match_id)
    minute_targets = compute_minute_targets(match_id)

    if not minutes_sorted:
        st.warning("No tracking frames found for this match within the first 100 minutes.")
        return

    st.markdown(
        f"Found tracking data for **{len(minutes_sorted)}** minutes "
        f"(from minute {min(minutes_sorted)} to {max(minutes_sorted)})."
    )

    # Optional: filter which minutes to display (e.g., up to a certain minute)
    max_display_minute = st.slider(
        "Show minutes up to",
        min_value=min(minutes_sorted),
        max_value=min(MAX_MINUTE - 1, max(minutes_sorted)),
        value=min(MAX_MINUTE - 1, max(minutes_sorted)),
    )

    minutes_to_show = [m for m in minutes_sorted if m <= max_display_minute]

    st.markdown("---")
    st.subheader("Minute grid")

    # 4 pitches per row
    cols_per_row = 4

    # Targets indexed by minute for quick lookup
    targets_map = {
        row["minute"]: row for _, row in minute_targets.iterrows()
    }

    # iterate over minutes in chunks of 4
    for i in range(0, len(minutes_to_show), cols_per_row):
        row_minutes = minutes_to_show[i : i + cols_per_row]
        cols = st.columns(len(row_minutes))
        for col, minute in zip(cols, row_minutes):
            with col:
                fig = plot_minute_pitch(minute, meta, minute_data)
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)

                    # Targets for this minute
                    tgt = targets_map.get(minute)
                    if tgt is not None:
                        shot_label = "Yes" if tgt["shot_next5"] else "No"
                        st.caption(
                            f"Next 5 mins (m→m+5]: Shot occurs: **{shot_label}**"
                        )
                else:
                    st.write(f"No tracking data for minute {minute}")

if __name__ == "__main__":
    main()
