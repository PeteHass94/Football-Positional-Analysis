import json
from pathlib import Path

import pandas as pd
import streamlit as st
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
# Data loaders (cached)
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
def load_phases_of_play(match_id: int) -> pd.DataFrame:
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_phases_of_play.csv"
    return pd.read_csv(path)


@st.cache_data
def load_tracking_sample(match_id: int, n_frames: int = 500) -> pd.DataFrame:
    """
    Load the first n_frames from the tracking_extrapolated jsonl file
    and flatten into a simple frame-level table.

    We don't explode all player positions here – this is just for
    exploration / sanity checking.
    """
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            # if i >= n_frames:
            #     break
            obj = json.loads(line)

            ball = obj.get("ball_data", {}) or {}
            poss = obj.get("possession", {}) or {}

            rows.append(
                {
                    "frame": obj.get("frame"),
                    "timestamp": obj.get("timestamp"),
                    "period": obj.get("period"),
                    "ball_x": ball.get("x"),
                    "ball_y": ball.get("y"),
                    "ball_z": ball.get("z"),
                    "ball_is_detected": ball.get("is_detected"),
                    "in_possession_player_id": poss.get("player_id"),
                    "in_possession_group": poss.get("group"),
                    "n_players": len(obj.get("player_data", [])),
                }
            )

    return pd.DataFrame(rows)

@st.cache_data
def load_all_frames(match_id: int) -> list:
    """
    Load ALL frames for the match, but keep each as a small raw dict.
    We do NOT explode player_data here — we keep the JSON structure intact.
    Good for interactive inspection with a slider.
    """
    path = BASE_MATCH_DIR / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"
    frames = []
    with path.open() as f:
        for line in f:
            frames.append(json.loads(line))
    return frames

@st.cache_data
def load_players_table(match_id: int) -> pd.DataFrame:
    """
    Flatten the 'players' section from the match JSON into a DataFrame,
    and add a team_name column for easy home/away filtering.
    """
    meta = load_match_metadata(match_id)
    players = meta.get("players", [])

    if not players:
        return pd.DataFrame()

    # Flatten nested structures, e.g. player_role, playing_time.total...
    df = pd.json_normalize(players, sep="_")

    # Map team_id -> team name
    team_map = {
        meta["home_team"]["id"]: meta["home_team"]["short_name"],
        meta["away_team"]["id"]: meta["away_team"]["short_name"],
    }
    df["team_name"] = df["team_id"].map(team_map)

    # Nice ordering
    order_cols = [
        "team_name",
        "team_id",
        "number",
        "short_name",
        "first_name",
        "last_name",
        "player_role_position_group",
        "player_role_name",
        "player_role_acronym",
        "goal",
        "own_goal",
        "yellow_card",
        "red_card",
        "playing_time_total_minutes_played",
        "playing_time_total_minutes_tip",
        "playing_time_total_minutes_otip",
        "playing_time_total_minutes_played_regular_time",
        "start_time",
        "end_time",
        "id",
        "team_player_id",
        "trackable_object",
        "birthday",
        "gender",
    ]
    existing_cols = [c for c in order_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining]

    return df



# ---------------------------------------------------------
# Column description helpers
# ---------------------------------------------------------

def build_common_docs() -> dict:
    # Base descriptions reused across datasets
    return {
        "index": "Row index within this file (0-based).",
        "match_id": "Identifier of the match.",
        "frame_start": "First tracking frame index associated with this row.",
        "frame_end": "Last tracking frame index associated with this row.",
        "frame_physical_start": "First frame index within the physical (on-pitch) action.",
        "time_start": "Start time of this row (hh:mm.ss within the match period).",
        "time_end": "End time of this row (hh:mm.ss within the match period).",
        "minute_start": "Start minute (integer) of this row.",
        "second_start": "Start second (integer) of this row.",
        "duration": "Duration of this row in seconds (approx).",
        "period": "Match period (1 = first half, 2 = second half, etc.).",
        "attacking_side_id": "Numeric code for the side currently attacking.",
        "attacking_side": "Text label for the side currently attacking (home / away).",
        "team_id": "Identifier of the team related to this row.",
        "team_shortname": "Short name / acronym of the team.",
        "x_start": "Start x-coordinate of the event/phase (pitch length axis, in metres).",
        "y_start": "Start y-coordinate of the event/phase (pitch width axis, in metres).",
        "x_end": "End x-coordinate of the event/phase.",
        "y_end": "End y-coordinate of the event/phase.",
        "channel_start": "Pitch vertical ‘channel’ at the start location (e.g. left / half-space / central).",
        "channel_end": "Pitch vertical ‘channel’ at the end location.",
        "third_start": "Pitch third at the start location (defensive / middle / attacking).",
        "third_end": "Pitch third at the end location.",
        "penalty_area_start": "Whether the start location is inside the penalty area.",
        "penalty_area_end": "Whether the end location is inside the penalty area.",
        "lead_to_shot": "Flag indicating that this row is in the build-up to a shot.",
        "lead_to_goal": "Flag indicating that this row is in the build-up to a goal.",
    }


def build_dynamic_events_docs(df: pd.DataFrame) -> pd.DataFrame:
    docs = build_common_docs()
    docs.update(
        {
            "event_id": "Unique identifier of the event (string key).",
            "event_type_id": "Numeric code for the event type.",
            "event_type": "Event type (e.g. pass, carry, duel, shot).",
            "player_id": "Identifier of the main player involved in the event.",
            "player_shortname": "Short name of the player linked to the event.",
            "player_in_possession_id": "Identifier of the player in possession at this moment.",
            "player_in_possession_shortname": "Short name of the player in possession.",
            "player_in_possession_position": "On-pitch position label for the player in possession.",
            "team_in_possession_id": "Team identifier for the team in possession.",
            "team_in_possession_shortname": "Short name of the team in possession.",
            "game_interruption_before": "Type of interruption immediately before this event (kick-off, foul, etc.).",
            "game_interruption_after": "Type of interruption immediately after this event.",
            "xg": "Expected Goals (if present) – model estimate of scoring probability for this event.",
            "xa": "Expected Assists (if present) – contribution to shot creation.",
            "xloss_player_possession_end": "SkillCorner metric of possession value loss/gain by the player at the end of the event.",
            "team_in_possession_width_start": "Width (max y - min y) of the team in possession at event start.",
            "team_in_possession_length_start": "Length (max x - min x) of the team in possession at event start.",
            "team_out_of_possession_width_start": "Width of the out-of-possession team at event start.",
            "team_out_of_possession_length_start": "Length of the out-of-possession team at event start.",
        }
    )

    rows = []
    for col in df.columns:
        desc = docs.get(col, "No description added yet – refer to SkillCorner documentation if needed.")
        rows.append({"column": col, "description": desc})

    return pd.DataFrame(rows)


def build_phases_docs(df: pd.DataFrame) -> pd.DataFrame:
    docs = build_common_docs()
    docs.update(
        {
            "team_in_possession_id": "Team identifier for the team in possession throughout the phase.",
            "team_in_possession_shortname": "Short name of the team in possession during the phase.",
            "n_player_possessions_in_phase": "Number of individual on-ball player possessions within this phase.",
            "phase_type_id": "Numeric code describing the phase type (e.g. build-up, sustained attack).",
            "phase_type": "Text label for the phase type.",
            "team_in_possession_width_start": "Width of the team in possession at the start of the phase.",
            "team_in_possession_width_end": "Width of the team in possession at the end of the phase.",
            "team_in_possession_length_start": "Length of the team in possession at the start of the phase.",
            "team_in_possession_length_end": "Length of the team in possession at the end of the phase.",
            "team_out_of_possession_width_start": "Width of the out-of-possession team at the start of the phase.",
            "team_out_of_possession_width_end": "Width of the out-of-possession team at the end of the phase.",
            "team_out_of_possession_length_start": "Length of the out-of-possession team at the start of the phase.",
            "team_out_of_possession_length_end": "Length of the out-of-possession team at the end of the phase.",
        }
    )

    rows = []
    for col in df.columns:
        desc = docs.get(col, "No description added yet – refer to SkillCorner documentation if needed.")
        rows.append({"column": col, "description": desc})

    return pd.DataFrame(rows)


def build_tracking_docs(df: pd.DataFrame) -> pd.DataFrame:
    docs = {
        "frame": "Sequential index of the tracking frame.",
        "timestamp": "Time within the match period (hh:mm:ss.ff) for this frame.",
        "period": "Match period (1 = first half, 2 = second half, etc.).",
        "ball_x": "Ball x-coordinate in pitch metres (length).",
        "ball_y": "Ball y-coordinate in pitch metres (width).",
        "ball_z": "Ball z-coordinate in metres (height).",
        "ball_is_detected": "True if the ball is detected in this frame, False otherwise.",
        "in_possession_player_id": "Identifier of the player considered in possession for this frame.",
        "in_possession_group": "Group label for possession (home / away / none).",
        "n_players": "Number of players with tracking data in this frame.",
    }

    rows = []
    for col in df.columns:
        desc = docs.get(col, "No description added yet – derived from tracking_extrapolated.jsonl.")
        rows.append({"column": col, "description": desc})

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Page layout
# ---------------------------------------------------------

def main():
    st.title("📊 Data exploration")

    st.markdown(
        """
        Explore the raw SkillCorner open data per match.  
        Use this page to sanity-check the inputs before building positional aggregates or models.
        """
    )

    # Match selector
    match_id = st.selectbox("Select a match", MATCH_IDS, index=0)

    # Load core files
    meta = load_match_metadata(match_id)
    dyn = load_dynamic_events(match_id)
    phases = load_phases_of_play(match_id)
    tracking_sample = load_tracking_sample(match_id)

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

    with st.expander("Show raw match metadata"):
        st.json(meta, expanded=False)

    st.markdown("---")

    # Tabs for the 3 core data files and additional frame viewer and players table
    tab_dyn, tab_phase, tab_track, tab_frames, tab_players = st.tabs(
        ["Dynamic events", "Phases of play", "Tracking (sample)", "Frame viewer", "Players"]
    )


    # -----------------------------------------------------
    # Dynamic events tab
    # -----------------------------------------------------
    with tab_dyn:
        st.subheader("Dynamic events")
        st.markdown(
            """
            One row per on-ball **event** (passes, carries, duels, shots, etc.),  
            enriched with context (phase, possession team, pitch zones, advanced metrics).
            """
        )

        st.write(f"Rows: **{len(dyn):,}**  •  Columns: **{dyn.shape[1]}**")

        st.markdown("**Preview**")
        st.dataframe(
            dyn,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Column descriptions**")
        docs_dyn = build_dynamic_events_docs(dyn)
        st.dataframe(
            docs_dyn,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # Phases of play tab
    # -----------------------------------------------------
    with tab_phase:
        st.subheader("Phases of play")
        st.markdown(
            """
            One row per **phase of play** – contiguous periods where the same team is in possession,
            with aggregated positional information for both teams.
            """
        )

        st.write(f"Rows: **{len(phases):,}**  •  Columns: **{phases.shape[1]}**")

        st.markdown("**Preview**")
        st.dataframe(
            phases,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Column descriptions**")
        docs_phase = build_phases_docs(phases)
        st.dataframe(
            docs_phase,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # Tracking tab
    # -----------------------------------------------------
    with tab_track:
        st.subheader("Tracking (extrapolated) – sample")
        st.markdown(
            """
            Frame-by-frame tracking data (`*_tracking_extrapolated.jsonl`).  
            This sample flattens the first N frames into a simple table that
            shows ball location, possession label, and number of tracked players.
            """
        )

        st.write(f"Sampled frames: **{len(tracking_sample):,}**")

        st.markdown("**Preview**")
        st.dataframe(
            tracking_sample,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Column descriptions**")
        docs_track = build_tracking_docs(tracking_sample)
        st.dataframe(
            docs_track,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # Frame viewer tab
    # -----------------------------------------------------
    
    with tab_frames:
        st.subheader("Frame viewer")

        st.markdown(
            """
            Explore an **individual tracking frame**.  
            This shows:  
            - Full `ball_data`  
            - Possession metadata  
            - Image corner projection  
            - Full `player_data` list (22 player entries)  
            """
        )

        # Load all frames (cached)
        frames = load_all_frames(match_id)
        n_frames = len(frames)

        st.write(f"Total frames: **{n_frames:,}**")

        # Slider to choose frame index
        frame_idx = st.slider(
            "Select frame index",
            min_value=0,
            max_value=n_frames - 1,
            value=0,
            step=1,
        )

        selected = frames[frame_idx]

        st.markdown("### Raw frame JSON")
        st.json(selected)

        # Extract player data into DataFrame for readability
        if "player_data" in selected:
            players_df = pd.DataFrame(selected["player_data"])
            st.markdown("### Player data table")
            st.dataframe(players_df, use_container_width=True)

    # -----------------------------------------------------
    # Players tab
    # -----------------------------------------------------
    
    with tab_players:
        st.subheader("Players")

        players_df = load_players_table(match_id)

        if players_df.empty:
            st.info("No player data found for this match.")
        else:
            home_team = meta["home_team"]["short_name"]
            away_team = meta["away_team"]["short_name"]
            home_id = meta["home_team"]["id"]
            away_id = meta["away_team"]["id"]

            home_df = players_df[players_df["team_id"] == home_id].copy()
            away_df = players_df[players_df["team_id"] == away_id].copy()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {home_team} – players")
                st.write(f"Count: **{len(home_df)}**")
                st.dataframe(
                    home_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "playing_time_by_period": st.column_config.JsonColumn(
                            "Playing time by period"
                            )
                    }
                )

            with col2:
                st.markdown(f"### {away_team} – players")
                st.write(f"Count: **{len(away_df)}**")
                st.dataframe(
                    away_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "playing_time_by_period": st.column_config.JsonColumn(
                            "Playing time by period"
                            )
                    }
                )

            st.markdown("---")
            st.markdown("### Player details")

            # Combined selector so you can inspect any player (home or away)
            players_df_display = players_df.copy()
            players_df_display["label"] = (
                players_df_display["team_name"]
                + " | #"
                + players_df_display["number"].astype(str)
                + " "
                + players_df_display["short_name"]
            )

            selected_label = st.selectbox(
                "Select a player to view full JSON",
                options=players_df_display["label"],
            )

            selected_row = players_df_display[
                players_df_display["label"] == selected_label
            ].iloc[0]

            # Find original raw dict (for exact nested structure)
            raw_players = load_match_metadata(match_id).get("players", [])
            raw_player = next(
                (p for p in raw_players if p["id"] == selected_row["id"]),
                None,
            )

            if raw_player is not None:
                st.json(raw_player)
            else:
                st.info("Could not find raw JSON for this player.")
    

if __name__ == "__main__":
    main()
