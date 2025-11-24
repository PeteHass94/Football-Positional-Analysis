import json
from pathlib import Path

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

# ---------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------

@st.cache_data
def load_match_metadata(match_id: int) -> dict:
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_match.json"
    with path.open() as f:
        return json.load(f)

@st.cache_data
def load_dynamic_events(match_id: int) -> pd.DataFrame:
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_dynamic_events.csv"
    return pd.read_csv(path)

@st.cache_data
def load_tracking_frames(match_id: int) -> list[dict]:
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"
    frames = []
    with path.open() as f:
        for line in f:
            frames.append(json.loads(line))
    return frames

@st.cache_data
def build_frame_index(match_id: int) -> dict[int, int]:
    frames = load_tracking_frames(match_id)
    return {f["frame"]: i for i, f in enumerate(frames)}

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
    return df[[
        "id", "team_id", "team_name", "number", "short_name",
        "player_role_position_group", "player_role_name", "player_role_acronym"
    ]]

# ---------------------------------------------------------
# Pitch Plotter
# ---------------------------------------------------------

def plot_frame_on_pitch(frame_dict, meta, player_lookup):

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=meta["pitch_length"],
        pitch_width=meta["pitch_width"],
        line_zorder=1
    )
    fig, ax = pitch.draw(figsize=(8, 6))

    player_map = {row["id"]: row for _, row in player_lookup.iterrows()}
    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]

    home_x, home_y = [], []
    away_x, away_y = [], []

    for p in frame_dict.get("player_data", []):
        pid, x, y = p.get("player_id"), p.get("x"), p.get("y")
        if pid not in player_map or x is None or y is None:
            continue
        if player_map[pid]["team_id"] == home_id:
            home_x.append(x); home_y.append(y)
        else:
            away_x.append(x); away_y.append(y)

    # Plot players with kit colours
    if home_x:
        pitch.scatter(
            home_x, home_y, ax=ax,
            facecolor=meta["home_team_kit"]["jersey_color"],
            edgecolor=meta["home_team_kit"]["number_color"],
            s=80,
            label=meta["home_team"]["short_name"],
        )
    if away_x:
        pitch.scatter(
            away_x, away_y, ax=ax,
            facecolor=meta["away_team_kit"]["jersey_color"],
            edgecolor=meta["away_team_kit"]["number_color"],
            s=80,
            label=meta["away_team"]["short_name"],
        )

    # Ball
    ball = frame_dict.get("ball_data", {})
    if ball.get("is_detected"):
        pitch.scatter(
            ball["x"], ball["y"], ax=ax,
            marker="football",
            facecolor="white",
            edgecolors="black",
            s=160,
            label="Ball",
        )

    ts = frame_dict.get("timestamp")
    ax.set_title(f"Frame {frame_dict['frame']} | Timestamp {ts}", fontsize=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=9)

    return fig

# ---------------------------------------------------------
# Streamlit Page
# ---------------------------------------------------------

def main():

    st.title("⚽ Match Visualisation")

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

    st.markdown("---")

    with st.expander("Show raw match metadata"):
        st.json(meta, expanded=False)

    dyn = load_dynamic_events(match_id)
    frames = load_tracking_frames(match_id)
    frame_index = build_frame_index(match_id)
    player_lookup = build_player_lookup(match_id)

    # -----------------------------------------------------
    # Session State Init / Reset when match changes
    # -----------------------------------------------------
    if "current_match_id" not in st.session_state:
        st.session_state.current_match_id = match_id

    # reset event/frame when match changes
    if st.session_state.current_match_id != match_id:
        st.session_state.current_match_id = match_id
        st.session_state.event_pos = 0
        st.session_state.frame_number = None

    if "event_pos" not in st.session_state:
        st.session_state.event_pos = 0
    if "frame_number" not in st.session_state:
        st.session_state.frame_number = None

    # Clamp event_pos to valid range (just in case)
    st.session_state.event_pos = max(0, min(st.session_state.event_pos, len(dyn) - 1))

    # -----------------------------------------------------
    # Event Navigation Buttons
    # -----------------------------------------------------

    st.markdown("### Dynamic Event Navigation")

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("⬅ Previous Event"):
            st.session_state.event_pos = max(0, st.session_state.event_pos - 1)
            st.session_state.frame_number = None  # reset within-event frame

    with col3:
        if st.button("Next Event ➡"):
            st.session_state.event_pos = min(len(dyn) - 1, st.session_state.event_pos + 1)
            st.session_state.frame_number = None

    # Selectbox synced by *position*
    event_positions = list(range(len(dyn)))
    selected_pos = st.selectbox(
        "Select event (by index)",
        options=event_positions,
        index=st.session_state.event_pos,
        key="event_select",
    )

    if selected_pos != st.session_state.event_pos:
        st.session_state.event_pos = selected_pos
        st.session_state.frame_number = None

    event = dyn.iloc[st.session_state.event_pos]

    st.markdown("---")
    st.subheader(f"Event Details (row {st.session_state.event_pos})")
    st.json(event.to_dict(), expanded=False)

    # -----------------------------------------------------
    # Frame Handling
    # -----------------------------------------------------

    if (
        hasattr(event, "frame_start") and hasattr(event, "frame_end")
        and not pd.isna(event.frame_start) and not pd.isna(event.frame_end)
    ):
        frame_start = int(event.frame_start)
        frame_end = int(event.frame_end)

        if st.session_state.frame_number is None:
            st.session_state.frame_number = frame_start

        st.markdown("### Frame Navigation (inside event)")

        c1, c2, c3 = st.columns([1, 3, 1])

        with c1:
            if st.button("◀ Previous Frame"):
                st.session_state.frame_number = max(frame_start, st.session_state.frame_number - 1)

        with c3:
            if st.button("Next Frame ▶"):
                st.session_state.frame_number = min(frame_end, st.session_state.frame_number + 1)

        if frame_start < frame_end:
            new_frame = st.slider(
                "Frame",
                frame_start,
                frame_end,
                value=st.session_state.frame_number,
                key="frame_slider",
            )
        else:
            new_frame = st.session_state.frame_number

        if new_frame != st.session_state.frame_number:
            st.session_state.frame_number = new_frame

        frame_num = st.session_state.frame_number
        st.write(f"Showing **frame {frame_num}**")

        if frame_num in frame_index:
            frame_dict = frames[frame_index[frame_num]]

            fig = plot_frame_on_pitch(frame_dict, meta, player_lookup)
            st.pyplot(fig, use_container_width=True)

            st.markdown("### Raw Frame JSON")
            st.json(frame_dict, expanded=False)
        else:
            st.error(f"Frame {frame_num} not found in tracking frames.")

    else:
        st.warning("This event has no valid frame_start / frame_end values.")

if __name__ == "__main__":
    main()
