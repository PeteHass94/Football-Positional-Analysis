import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
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

MAX_MINUTE = 100


# ---------------------------------------------------------
# Loaders
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
def build_frame_index(match_id: int) -> dict[int, int]:
    frames = load_tracking_frames(match_id)
    return {fr["frame"]: i for i, fr in enumerate(frames)}


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
# Helper: match label for selectbox
# ---------------------------------------------------------

def build_match_label(match_id: int) -> str:
    meta = load_match_metadata(match_id)
    home = meta["home_team"]["short_name"]
    away = meta["away_team"]["short_name"]
    score = f"{meta['home_team_score']}–{meta['away_team_score']}"
    date = str(meta["date_time"]).split("T")[0]
    return f"{match_id} : {home} vs {away} — {score} ({date})"


# ---------------------------------------------------------
# Minute aggregation (for average shapes)
# ---------------------------------------------------------

def timestamp_to_minute(ts: str) -> int:
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


@st.cache_data
def aggregate_minute_positions_with_roles(match_id: int):
    """
    Aggregate average positions per minute for home/away players and ball,
    including position_group for each player.
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

        ball = fr.get("ball_data") or {}
        if ball.get("is_detected") and ball.get("x") is not None and ball.get("y") is not None:
            b_acc = minute_ball_acc[minute]
            b_acc[0] += ball["x"]
            b_acc[1] += ball["y"]
            b_acc[2] += 1

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
                "position_group": info["player_role_position_group"],
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
# Standardisation helpers (attack → right per team panel)
# ---------------------------------------------------------

def standardise_x_for_team(x: float, is_away_panel: bool) -> float:
    """
    Simple panel-level standardisation:
    - Home panel: keep x as is (defend left, attack right in raw coords)
    - Away panel: mirror horizontally so away also attacks to the right
    """
    return x if is_away_panel else -x

def standardise_y_for_team(y: float, is_away_panel: bool) -> float:
    """
    Simple panel-level standardisation:
    - Home panel: keep y as is (defend left, attack right in raw coords)
    - Away panel: mirror horizontally so away also attacks to the right
    """
    return y if is_away_panel else -y

def standardise_xy_for_team(x: float, y: float, is_away_panel: bool):
    return standardise_x_for_team(x, is_away_panel), standardise_y_for_team(y, is_away_panel)

# ---------------------------------------------------------
# Plot average minute shapes with optional highlighting
# ---------------------------------------------------------

def plot_minute_team_panel(meta: dict,
                           minute_data: dict,
                           minute: int,
                           team_role: str,
                           highlight_group: str | None = None):
    """
    Plot average positions for a single team (home/away) in a given minute.
    highlight_group: position_group to highlight (or None / 'All' for no special highlight).
    """
    data = minute_data.get(minute)
    if data is None:
        return None

    is_away_panel = team_role == "away"
    points = data["away"] if is_away_panel else data["home"]
    ball = data["ball"]

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=meta["pitch_length"],
        pitch_width=meta["pitch_width"],
        line_zorder=1,
    )
    fig, ax = pitch.draw(figsize=(4, 3))

    # Plot all players low-key
    if points:
        xs_all = []
        ys_all = []
        for p in points:
            x_std, y_std = standardise_xy_for_team(p["x"], p["y"], is_away_panel)
            xs_all.append(x_std)
            ys_all.append(y_std)

        team_kit = meta["away_team_kit"] if is_away_panel else meta["home_team_kit"]
        team_name = meta["away_team"]["short_name"] if is_away_panel else meta["home_team"]["short_name"]

        pitch.scatter(
            xs_all,
            ys_all,
            ax=ax,
            facecolor=team_kit["jersey_color"],
            edgecolor="black" if team_kit["number_color"] == '#ffffff' else team_kit["number_color"],
            s=50,
            # alpha=0.75,
            label=team_name,
        )

        # Highlight selected position group
        if highlight_group and highlight_group != "All":
            xs_h = []
            ys_h = []
            labels = []
            for p in points:
                if p["position_group"] == highlight_group:
                    x_std, y_std = standardise_xy_for_team(p["x"], p["y"], is_away_panel)
                    xs_h.append(x_std)
                    ys_h.append(y_std)
                    labels.append(p["number"])

            if xs_h:
                pitch.scatter(
                    xs_h,
                    ys_h,
                    ax=ax,
                    facecolor="yellow",
                    edgecolor="black",
                    s=120,
                    zorder=4,
                    label=f"{highlight_group}",
                )
                # Annotate shirt numbers
                for x, y, num in zip(xs_h, ys_h, labels):
                    ax.text(x, y, str(num), ha="center", va="center", fontsize=8, weight="bold")

    # Ball
    if ball is not None and ball.get("x") is not None and ball.get("y") is not None:
        bx, by = standardise_xy_for_team(ball["x"], ball["y"], is_away_panel)
        pitch.scatter(
            bx,
            by,
            ax=ax,
            marker="football",
            facecolor="white",
            edgecolors="black",
            s=120,
            zorder=5,
        )

    team_label = meta["away_team"]["short_name"] if is_away_panel else meta["home_team"]["short_name"]
    ax.set_title(f"{team_label} — Minute {minute}", fontsize=10)
    return fig


# ---------------------------------------------------------
# Shot events & frame plotting
# ---------------------------------------------------------

def is_shot_event(row: pd.Series) -> bool:
    end_type = row.get("end_type")
    if pd.isna(end_type):
        return False
    return str(end_type).strip().lower() == "shot"


@st.cache_data
def get_shot_events(match_id: int) -> pd.DataFrame:
    dyn = load_dynamic_events(match_id).copy()
    # Preserve original index for reference
    dyn = dyn.reset_index().rename(columns={"index": "orig_index"})
    dyn["is_shot"] = dyn.apply(is_shot_event, axis=1)
    shots = dyn[dyn["is_shot"]].copy()
    return shots


def plot_shot_frame_for_team(meta: dict,
                             frame_dict: dict,
                             player_lookup: pd.DataFrame,
                             focus_team_role: str,
                             shot_team_id: int | None):
    """
    Plot a single frame for either home or away panel, with optional shot line.
    """
    is_away_panel = focus_team_role == "away"
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=meta["pitch_length"],
        pitch_width=meta["pitch_width"],
        line_zorder=1,
    )
    fig, ax = pitch.draw(figsize=(4, 3))

    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]
    team_kit_home = meta["home_team_kit"]
    team_kit_away = meta["away_team_kit"]
    team_short_home = meta["home_team"]["short_name"]
    team_short_away = meta["away_team"]["short_name"]

    player_map = {row["id"]: row for _, row in player_lookup.iterrows()}

    # Plot players
    home_x, home_y = [], []
    away_x, away_y = [], []
    for p in frame_dict.get("player_data", []):
        pid = p.get("player_id")
        x = p.get("x")
        y = p.get("y")
        if pid is None or x is None or y is None:
            continue
        info = player_map.get(pid)
        if info is None:
            continue
        if info["team_id"] == home_id:
            xs, ys = standardise_xy_for_team(x, y, is_away_panel=False)
            home_x.append(xs)
            home_y.append(ys)
        elif info["team_id"] == away_id:
            xs, ys = standardise_xy_for_team(x, y, is_away_panel=True)
            away_x.append(xs)
            away_y.append(ys)

    # Decide which set to show in this panel
    if not is_away_panel:
        if home_x:
            pitch.scatter(
                home_x,
                home_y,
                ax=ax,
                facecolor=team_kit_home["jersey_color"],
                edgecolor="black" if team_kit_home["number_color"] == '#ffffff' else team_kit_home["number_color"],
                s=70,
                label=team_short_home,
            )
    else:
        if away_x:
            pitch.scatter(
                away_x,
                away_y,
                ax=ax,
                facecolor=team_kit_away["jersey_color"],
                edgecolor="black" if team_kit_away["number_color"] == '#ffffff' else team_kit_away["number_color"],
                s=70,
                label=team_short_away,
            )

    # Ball + shot line
    ball = frame_dict.get("ball_data") or {}
    if ball.get("is_detected") and ball.get("x") is not None and ball.get("y") is not None:
        bx_raw, by_raw = ball["x"], ball["y"]
        bx_home, by_home = standardise_xy_for_team(bx_raw, by_raw, is_away_panel=False)
        bx_away, by_away = standardise_xy_for_team(bx_raw, by_raw, is_away_panel=True)

        if not is_away_panel:
            bx, by = bx_home, by_home
        else:
            bx, by = bx_away, by_away

        pitch.scatter(
            bx,
            by,
            ax=ax,
            marker="football",
            facecolor="white",
            edgecolors="black",
            s=140,
            zorder=5,
        )

        # Draw line from ball to goal for the shooting team panel only
        if shot_team_id is not None:
            focus_team_id = meta["away_team"]["id"] if is_away_panel else meta["home_team"]["id"]
            if focus_team_id == shot_team_id:
                # Right-hand goal in this standardised view
                goal_x = meta["pitch_length"] / 2
                goal_y = 0.0
                ax.plot(
                    [bx, goal_x],
                    [by, goal_y],
                    linestyle="--",
                    linewidth=2,
                    color="red",
                    alpha=0.8,
                )

    team_label = team_short_away if is_away_panel else team_short_home
    ts = frame_dict.get("timestamp")
    ax.set_title(f"{team_label} — frame {frame_dict['frame']} ({ts})", fontsize=9)
    return fig


# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------

def main():
    st.title("📐 Directional Shapes & Shot Frames")

    st.markdown(
        """
        This page explores three ideas:
        
        1. **Direction of play** – average minute shapes with each team always attacking → right  
        2. **Shot phases** – scrubbing through the frames of the possessions that end in shots  
        3. **Position groups** – highlighting specific roles inside those average minute shapes  
        """
    )

    # Match selector with rich labels
    match_labels = {build_match_label(mid): mid for mid in MATCH_IDS}
    label_selected = st.selectbox(
        "Select a match",
        options=list(match_labels.keys()),
        index=0,
    )
    match_id = match_labels[label_selected]

    meta = load_match_metadata(match_id)
    frames = load_tracking_frames(match_id)
    frame_index = build_frame_index(match_id)
    dyn = load_dynamic_events(match_id)
    player_lookup = build_player_lookup(match_id)

    home_short = meta["home_team"]["short_name"]
    away_short = meta["away_team"]["short_name"]

    st.markdown(
        f"**Match:** {home_short} vs {away_short} — "
        f"{meta['competition_edition']['competition']['name']} "
        f"({meta['competition_edition']['season']['name']})"
    )

    # Pre-compute minute aggregates (used in Sections 1 and 3)
    minute_data, minutes_sorted = aggregate_minute_positions_with_roles(match_id)

    st.markdown("---")

    # -----------------------------------------------------
    # 1) Average minute shapes (direction fixed)
    # -----------------------------------------------------
    st.header("1. Average minute shapes (attacking → right)")

    if not minutes_sorted:
        st.warning("No minute-level tracking aggregation available.")
    else:
        min_minute = min(minutes_sorted)
        max_minute = max(minutes_sorted)
        minute_sel_simple = st.slider(
            "Select minute of match (for average shapes)",
            min_value=min_minute,
            max_value=max_minute,
            value=min_minute,
            key="minute_shapes",
        )

        st.caption(
            "Average player locations in a given minute. "
            "Each team is shown in its own panel, always attacking to the right."
        )

        col_home1, col_away1 = st.columns(2)
        with col_home1:
            fig_h = plot_minute_team_panel(
                meta, minute_data, minute_sel_simple, team_role="home", highlight_group=None
            )
            if fig_h is not None:
                st.pyplot(fig_h, use_container_width=True)
            else:
                st.info("No home-team data for this minute.")

        with col_away1:
            fig_a = plot_minute_team_panel(
                meta, minute_data, minute_sel_simple, team_role="away", highlight_group=None
            )
            if fig_a is not None:
                st.pyplot(fig_a, use_container_width=True)
            else:
                st.info("No away-team data for this minute.")

    st.markdown("---")

    # -----------------------------------------------------
    # 2) Shot phases: frames from frame_start → frame_end
    # -----------------------------------------------------
    st.header("2. Shot phases: frames leading to the shot")

    shots = get_shot_events(match_id)
    
    if shots.empty:
        st.info("No events with end_type = 'shot' in this match.")
    else:
        team_col = "team_in_possession_shortname" if "team_in_possession_shortname" in shots.columns else (
            "team_in_possession_name" if "team_in_possession_name" in shots.columns else None
        )
        team_id_col = "team_in_possession_id" if "team_in_possession_id" in shots.columns else None

        def shot_label(pos: int) -> str:
            row = shots.iloc[pos]
            idx = int(row["orig_index"])
            etype = row.get("event_type", row.get("event_type_name", "shot"))
            t_start = row.get("time_start", row.get("second_start", ""))
            t_end = row.get("time_end", row.get("second_end", ""))
            f_start = row.get("frame_start", "")
            f_end = row.get("frame_end", "")
            team_str = row.get(team_col, "") if team_col else ""
            lead_goal = row.get("lead_to_goal", False)
            lg_str = " (led to goal)" if bool(lead_goal) else ""
            team = row.get("team_shortname", "")
            return (
                f"Phase {idx} : {team_str} {etype}{lg_str} | "
                f"Frames {f_start}–{f_end} ({t_start}→{t_end}) | "
                f"{team} shot"
            )

        shot_positions = list(range(len(shots)))
        shot_pos_sel = st.selectbox(
            "Select a shot phase (dynamic event with end_type='shot')",
            options=shot_positions,
            index=0,
            format_func=shot_label,
        )

        shot_row = shots.iloc[shot_pos_sel]

        # Frame range for this phase
        frame_start = None
        frame_end = None
        if "frame_start" in shot_row and not pd.isna(shot_row["frame_start"]):
            frame_start = int(shot_row["frame_start"])
        if "frame_end" in shot_row and not pd.isna(shot_row["frame_end"]):
            frame_end = int(shot_row["frame_end"])

        if frame_start is None or frame_end is None or frame_end < frame_start:
            st.error("Shot event has invalid frame_start/frame_end.")
        else:
            st.caption(
                "Scrub through the frames of this possession phase. "
                "Panels are standardised so each team attacks to the right; "
                "a red dashed line shows ball → goal for the shooting team."
            )

            frame_sel = st.slider(
                "Frame within shot phase",
                min_value=frame_start,
                max_value=frame_end,
                value=frame_end,
                key="shot_phase_frame",
            )

            if frame_sel not in frame_index:
                st.error(f"Frame {frame_sel} not found in tracking data.")
            else:
                frame_dict = frames[frame_index[frame_sel]]

                # Determine shooting team id (if available)
                shot_team_id = None
                if team_id_col and team_id_col in shot_row:
                    try:
                        shot_team_id = int(shot_row[team_id_col])
                    except Exception:
                        shot_team_id = None

                st.subheader(
                    f"Frame {frame_sel} at timestamp {frame_dict.get('timestamp')} "
                    f"(frames {frame_start}–{frame_end})"
                )

                col_h2, col_a2 = st.columns(2)
                with col_h2:
                    fig_sh_home = plot_shot_frame_for_team(
                        meta, frame_dict, player_lookup, focus_team_role="home", shot_team_id=shot_team_id
                    )
                    st.pyplot(fig_sh_home, use_container_width=True)

                with col_a2:
                    fig_sh_away = plot_shot_frame_for_team(
                        meta, frame_dict, player_lookup, focus_team_role="away", shot_team_id=shot_team_id
                    )
                    st.pyplot(fig_sh_away, use_container_width=True)

                with st.expander("Show raw shot event JSON"):
                    st.json(shot_row.to_dict(), expanded=False)

                with st.expander("Show raw frame JSON"):
                    st.json(frame_dict, expanded=False)

    st.markdown("---")

    # -----------------------------------------------------
    # 3) Average minute shapes with position-group highlight
    # -----------------------------------------------------
    st.header("3. Position groups inside average shapes")

    if not minutes_sorted:
        st.warning("No minute-level tracking aggregation available.")
    else:
        min_minute = min(minutes_sorted)
        max_minute = max(minutes_sorted)
        minute_sel = st.slider(
            "Select minute of match (for role highlight)",
            min_value=min_minute,
            max_value=max_minute,
            value=min_minute,
            key="minute_roles",
        )

        pos_groups = (
            player_lookup["player_role_position_group"]
            .dropna()
            .unique()
            .tolist()
        )
        pos_groups = sorted(pos_groups)
        pos_group_sel = st.selectbox(
            "Highlight position group",
            options=["All"] + pos_groups,
            index=0,
        )

        st.caption(
            "Same average shapes as in Section 1, but now you can pick a position group "
            "(e.g. Center Forward, Midfield, Full Back). Those players are highlighted in yellow."
        )

        col_home3, col_away3 = st.columns(2)
        with col_home3:
            fig_h = plot_minute_team_panel(
                meta, minute_data, minute_sel, team_role="home", highlight_group=pos_group_sel
            )
            if fig_h is not None:
                st.pyplot(fig_h, use_container_width=True)
            else:
                st.info("No home-team data for this minute.")

        with col_away3:
            fig_a = plot_minute_team_panel(
                meta, minute_data, minute_sel, team_role="away", highlight_group=pos_group_sel
            )
            if fig_a is not None:
                st.pyplot(fig_a, use_container_width=True)
            else:
                st.info("No away-team data for this minute.")


if __name__ == "__main__":
    main()
