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
def load_phases_of_play(match_id: int) -> pd.DataFrame:
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_phases_of_play.csv"
    return pd.read_csv(path)

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
    """
    Draw a SkillCorner pitch with players + ball for a single frame.
    Uses kit colours from match metadata.
    """
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

    st.title("📐 Phases of Play")

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
    
    phases = load_phases_of_play(match_id)
    dyn = load_dynamic_events(match_id)
    frames = load_tracking_frames(match_id)
    frame_index = build_frame_index(match_id)
    player_lookup = build_player_lookup(match_id)

    # -----------------------------------------------------
    # Session State Init / Reset when match changes
    # -----------------------------------------------------
    if "phases_match_id" not in st.session_state:
        st.session_state.phases_match_id = match_id

    if st.session_state.phases_match_id != match_id:
        st.session_state.phases_match_id = match_id
        st.session_state.phase_pos = 0
        st.session_state.phase_frame_number = None
        st.session_state.phase_event_pos = 0
        st.session_state.phase_event_frame_number = None

    if "phase_pos" not in st.session_state:
        st.session_state.phase_pos = 0
    if "phase_frame_number" not in st.session_state:
        st.session_state.phase_frame_number = None
    if "phase_event_pos" not in st.session_state:
        st.session_state.phase_event_pos = 0
    if "phase_event_frame_number" not in st.session_state:
        st.session_state.phase_event_frame_number = None

    # Clamp phase_pos range
    st.session_state.phase_pos = max(0, min(st.session_state.phase_pos, len(phases) - 1))

    # -----------------------------------------------------
    # Phase Navigation
    # -----------------------------------------------------

    st.markdown("### Phase navigation")

    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("⬅ Previous phase"):
            st.session_state.phase_pos = max(0, st.session_state.phase_pos - 1)
            st.session_state.phase_frame_number = None
            st.session_state.phase_event_pos = 0
            st.session_state.phase_event_frame_number = None
    with c3:
        if st.button("Next phase ➡"):
            st.session_state.phase_pos = min(len(phases) - 1, st.session_state.phase_pos + 1)
            st.session_state.phase_frame_number = None
            st.session_state.phase_event_pos = 0
            st.session_state.phase_event_frame_number = None

    # -----------------------------------------------------
    # Improved Phase Selector With Details
    # -----------------------------------------------------

    # Build labels like:
    # "12: WEL | Shot=False | Goal=True"

    def build_phase_label(idx, row):
        team = row.get("team_in_possession_shortname", "UNK")
        shot = bool(row.get("team_possession_lead_to_shot", False))
        goal = bool(row.get("team_possession_lead_to_goal", False))
        return f"{idx}: {team} | Shot={shot} | Goal={goal}"

    phase_positions = list(range(len(phases)))

    phase_labels = [
        build_phase_label(i, phases.iloc[i])
        for i in phase_positions
    ]

    selected_phase_label = st.selectbox(
        "Select phase",
        options=phase_labels,
        index=st.session_state.phase_pos,
        key="phase_select",
    )

    # Convert label back into index:
    selected_phase_pos = phase_labels.index(selected_phase_label)

    if selected_phase_pos != st.session_state.phase_pos:
        st.session_state.phase_pos = selected_phase_pos
        st.session_state.phase_frame_number = None
        st.session_state.phase_event_pos = 0
        st.session_state.phase_event_frame_number = None

    phase = phases.iloc[st.session_state.phase_pos]

    st.markdown("---")
    st.subheader(f"Phase details (row {st.session_state.phase_pos})")

    # Show key summary above the JSON
    cols = st.columns(5)
    with cols[0]:
        st.metric("Team in possession", phase.get("team_in_possession_shortname", ""))
    with cols[1]:
        st.metric("Period", phase.get("period", ""))
    with cols[2]:
        st.metric("Duration (s)", f"{phase.get('duration', 0):.1f}")
    with cols[3]:
        st.metric("Frames", f"{int(phase.frame_start)} → {int(phase.frame_end)}")
    with cols[4]:
        st.metric("Phase type", phase.get("team_in_possession_phase_type", ""))

    with st.expander("Show full phase JSON"):
        st.json(phase.to_dict(), expanded=False)
        
    # -----------------------------------------------------
    # Dynamic events within phase
    # -----------------------------------------------------

    st.subheader("Dynamic events in this phase")

    # Use frames to define belonging: event starts inside phase frame window
    events_in_phase = dyn[
        (dyn["frame_start"] >= phase.frame_start) & (dyn["frame_start"] <= phase.frame_end)
    ].copy()

    if events_in_phase.empty:
        st.info("No dynamic events found in this phase.")
        return

    # Small summary table
    display_cols = [
        c for c in [
            "event_id", "event_type", "team_in_possession_shortname",
            "time_start", "time_end", "frame_start", "frame_end"
        ] if c in events_in_phase.columns
    ]
    st.dataframe(
        events_in_phase[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # Frames within phase
    # -----------------------------------------------------

    if not pd.isna(phase.frame_start) and not pd.isna(phase.frame_end):

        frame_start = int(phase.frame_start)
        frame_end = int(phase.frame_end)

        if st.session_state.phase_frame_number is None:
            st.session_state.phase_frame_number = frame_start

        st.markdown("### Frames in this phase")

        fc1, _, fc3 = st.columns([1, 3, 1])
        with fc1:
            if st.button("◀ Previous frame (phase)"):
                st.session_state.phase_frame_number = max(frame_start, st.session_state.phase_frame_number - 1)
        with fc3:
            if st.button("Next frame (phase) ▶"):
                st.session_state.phase_frame_number = min(frame_end, st.session_state.phase_frame_number + 1)

        if frame_start < frame_end:
            new_phase_frame = st.slider(
                "Frame in phase",
                frame_start,
                frame_end,
                value=st.session_state.phase_frame_number,
                key="phase_frame_slider",
            )
        else:
            new_phase_frame = st.session_state.phase_frame_number

        if new_phase_frame != st.session_state.phase_frame_number:
            st.session_state.phase_frame_number = new_phase_frame

        phase_frame_num = st.session_state.phase_frame_number
        st.write(f"Showing **frame {phase_frame_num}** for this phase")

        if phase_frame_num in frame_index:
            phase_frame_dict = frames[frame_index[phase_frame_num]]
            fig_phase = plot_frame_on_pitch(phase_frame_dict, meta, player_lookup)
            st.pyplot(fig_phase, use_container_width=True)

            st.markdown("**Raw frame JSON (phase frame)**")
            st.json(phase_frame_dict, expanded=False)
        else:
            st.error(f"Frame {phase_frame_num} not found in tracking frames.")
    else:
        st.warning("Phase has no valid frame_start / frame_end.")

    st.markdown("---")


    # --- Event navigation inside phase ---

    st.markdown("### Inspect a dynamic event in this phase")

    # Positions within the filtered subset
    n_events = len(events_in_phase)
    event_positions = list(range(n_events))

    # Clamp
    st.session_state.phase_event_pos = max(0, min(st.session_state.phase_event_pos, n_events - 1))

    ec1, _, ec3 = st.columns([1, 3, 1])
    with ec1:
        if st.button("⬅ Previous event (in phase)"):
            st.session_state.phase_event_pos = max(0, st.session_state.phase_event_pos - 1)
            st.session_state.phase_event_frame_number = None
    with ec3:
        if st.button("Next event (in phase) ➡"):
            st.session_state.phase_event_pos = min(n_events - 1, st.session_state.phase_event_pos + 1)
            st.session_state.phase_event_frame_number = None

    selected_event_pos = st.selectbox(
        "Select event in this phase (by position)",
        options=event_positions,
        index=st.session_state.phase_event_pos,
        key="phase_event_select",
    )
    if selected_event_pos != st.session_state.phase_event_pos:
        st.session_state.phase_event_pos = selected_event_pos
        st.session_state.phase_event_frame_number = None

    # Get the corresponding event row (original index preserved)
    event_row = events_in_phase.iloc[st.session_state.phase_event_pos]

    st.markdown(f"**Event details (within phase, position {st.session_state.phase_event_pos})**")
    st.json(event_row.to_dict(), expanded=False)

    # Frames for this event
    if not pd.isna(event_row.frame_start) and not pd.isna(event_row.frame_end):
        e_frame_start = int(event_row.frame_start)
        e_frame_end = int(event_row.frame_end)

        if st.session_state.phase_event_frame_number is None:
            st.session_state.phase_event_frame_number = e_frame_start

        st.markdown("#### Frames for this dynamic event")

        ef1, _, ef3 = st.columns([1, 3, 1])
        with ef1:
            if st.button("◀ Previous frame (event)"):
                st.session_state.phase_event_frame_number = max(
                    e_frame_start, st.session_state.phase_event_frame_number - 1
                )
        with ef3:
            if st.button("Next frame (event) ▶"):
                st.session_state.phase_event_frame_number = min(
                    e_frame_end, st.session_state.phase_event_frame_number + 1
                )

        if e_frame_start < e_frame_end:
            new_event_frame = st.slider(
                "Frame in event",
                e_frame_start,
                e_frame_end,
                value=st.session_state.phase_event_frame_number,
                key="phase_event_frame_slider",
            )
        else:
            new_event_frame = st.session_state.phase_event_frame_number

        if new_event_frame != st.session_state.phase_event_frame_number:
            st.session_state.phase_event_frame_number = new_event_frame

        evt_frame_num = st.session_state.phase_event_frame_number
        st.write(f"Showing **frame {evt_frame_num}** for this dynamic event")

        if evt_frame_num in frame_index:
            evt_frame_dict = frames[frame_index[evt_frame_num]]
            fig_evt = plot_frame_on_pitch(evt_frame_dict, meta, player_lookup)
            st.pyplot(fig_evt, use_container_width=True)

            st.markdown("**Raw frame JSON (event frame)**")
            st.json(evt_frame_dict, expanded=False)
        else:
            st.error(f"Frame {evt_frame_num} not found in tracking frames.")
    else:
        st.warning("Selected event has no valid frame_start / frame_end.")

if __name__ == "__main__":
    main()
