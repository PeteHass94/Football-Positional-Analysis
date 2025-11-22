import streamlit as st
import pandas as pd
from supabase import create_client
from utils.page_components import add_common_page_elements
from utils.extractors.data_fetcher import fetch_lineups, fetch_managers, fetch_incidents, fetch_shotmap

# ------------------ Supabase ------------------
supabase = create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])

# ------------------ Cached lookups ------------------
@st.cache_data(show_spinner=False)
def get_fixtures():
    res = supabase.table("fixtures").select(
        "id, fixture_id, home_team_id, away_team_id, season_id, round"
    ).execute()
    return res.data or []

@st.cache_data(show_spinner=False)
def get_teams():
    res = supabase.table("teams").select("id, team_id, name").execute()
    return res.data or []

@st.cache_data(show_spinner=False)
def get_existing_players_ids() -> set:
    res = supabase.table("players").select("player_id").execute()
    return {r["player_id"] for r in (res.data or []) if r.get("player_id") is not None}

@st.cache_data(show_spinner=False)
def get_existing_managers_ids() -> set:
    res = supabase.table("managers").select("manager_id, team_id").execute()
    return {r["manager_id"] for r in (res.data or []) if r.get("manager_id") is not None}

# ------------------ Utilities ------------------
def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _mk_match_minute(minute, added):
    if minute is None:
        return None
    return int(minute) + (int(added) if added else 0)

def _team_id_from_incident(inc):
    team = (inc or {}).get("team")
    if isinstance(team, dict):
        return team.get("id")
    if isinstance(team, int):
        return team
    if inc.get("isHome"):
        return inc.get("homeTeam", {}).get("id")
    return inc.get("awayTeam", {}).get("id")

# ------------------ Existing rows (per fixture) ------------------
def get_existing_pf_for_fixture(fixture_id: int) -> set:
    res = supabase.table("players_fixtures").select("player_id, team_id").eq("fixture_id", fixture_id).execute()
    return {(r.get("player_id"), r.get("team_id")) for r in (res.data or [])}

def get_existing_cards_for_fixture(fixture_id: int) -> set:
    res = supabase.table("cards").select(
        "player_id,team_id,card_minute,added_time,yellow,red,yellow_2"
    ).eq("fixture_id", fixture_id).execute()
    rows = res.data or []
    return {
        (
            r.get("player_id"),
            r.get("team_id"),
            r.get("card_minute"),
            r.get("added_time"),
            bool(r.get("yellow")),
            bool(r.get("yellow_2")),
            bool(r.get("red")),
        )
        for r in rows
    }

def get_existing_goals_for_fixture(fixture_id: int) -> set:
    res = supabase.table("goals").select(
        "player_id,team_id,goal_minute,added_time,type,is_own_goal"
    ).eq("fixture_id", fixture_id).execute()
    rows = res.data or []
    return {
        (
            r.get("player_id"),
            r.get("team_id"),
            r.get("goal_minute"),
            r.get("added_time"),
            r.get("type"),
            bool(r.get("is_own_goal")),
        )
        for r in rows
    }

def get_existing_shots_for_fixture(fixture_id: int) -> tuple[set, set]:
    res = supabase.table("shots").select(
        "shot_id,player_id,team_id,minute,added_time,situation"
    ).eq("fixture_id", fixture_id).execute()
    rows = res.data or []
    have_by_id = {r["shot_id"] for r in rows if r.get("shot_id") is not None}
    have_by_fallback = {
        (r.get("player_id"), r.get("team_id"), r.get("minute"), r.get("added_time"), r.get("situation"))
        for r in rows if r.get("shot_id") is None
    }
    return have_by_id, have_by_fallback

# ------------------ Parsers ------------------
def parse_players_from_lineups(lineups_json, home_team_id: int | None, away_team_id: int | None):
    """
    Build rows for:
      - public.players
      - public.players_fixtures

    Notes:
    - Uses the SELECTED side's team_id (home_team_id / away_team_id) as the authoritative team
      to keep it consistent with your fixtures table.
    - minutes_played comes from statistics.minutesPlayed when present.
    - substituted_on/off + times are left None here; we enrich later from incidents.
    """
    def _safe_int(x):
        try:
            return int(x) if x is not None else None
        except Exception:
            return None

    players_rows: list[dict] = []
    pf_rows: list[dict] = []

    if not isinstance(lineups_json, dict):
        return players_rows, pf_rows

    for side_key, side_team_id in (("home", home_team_id), ("away", away_team_id)):
        side = lineups_json.get(side_key) or {}
        for entry in side.get("players", []) or []:
            p = (entry or {}).get("player") or {}
            pid = p.get("id")
            if not pid:
                continue

            # ---- players (global) ----
            players_rows.append(
                {
                    "player_id": _safe_int(pid),
                    "name": p.get("name"),
                    "short_name": p.get("shortName"),
                    "dateOfBirthTimestamp": _safe_int(p.get("dateOfBirthTimestamp")),
                    # If you want to trust the per-entry teamId instead, use:
                    # "team_id": _safe_int(entry.get("teamId")),
                    "team_id": _safe_int(side_team_id),
                }
            )

            # ---- players_fixtures (per fixture) ----
            stats = (entry or {}).get("statistics") or {}
            minutes_played = _safe_int(stats.get("minutesPlayed"))

            started = not bool(entry.get("substitute"))
            substitute = bool(entry.get("substitute"))

            pf_rows.append(
                {
                    "player_id": _safe_int(pid),
                    "fixture_id": None,             # filled by caller
                    "team_id": _safe_int(side_team_id),
                    "started": started,
                    "substitute": substitute,
                    "substituted_on": None,         # enriched later from incidents
                    "substituted_off": None,        # enriched later from incidents
                    "minutes_played": minutes_played,
                    "subbed_on_time": None,
                    "subbed_off_time": None,
                    "game_minutes_played": minutes_played,
                }
            )

    return players_rows, pf_rows

from typing import Any, Dict, List, Optional

def parse_managers(managers_json: dict, home_team_id: int | None, away_team_id: int | None):
    """
    Convert a SofaScore-style managers payload into rows for your `managers` table.

    Input example:
      {
        "homeManager": {"name":"Fabian Hurzeler","slug":"fabian-hurzeler","shortName":"F. Hurzeler","id":788529,...},
        "awayManager": {"name":"Thomas Frank","slug":"thomas-frank","shortName":"T. Frank","id":94249,...}
      }

    Returns:
      dict with optional home/away rows ready for insert (no 'id' yet):
      {
        "home": {"name":..., "short_name":..., "manager_id":..., "slug":..., "team_id": home_team_id},
        "away": {"name":..., "short_name":..., "manager_id":..., "slug":..., "team_id": away_team_id}
      }
    """
    def _row(mgr: dict | None, team_id: int | None):
        if not mgr:
            return None
        return {
            "name": mgr.get("name"),
            "short_name": mgr.get("shortName"),
            "manager_id": mgr.get("id"),   # external SofaScore id
            "slug": mgr.get("slug"),
            "team_id": team_id,            # required by schema
        }

    home_row = _row(managers_json.get("homeManager"), home_team_id) if managers_json else None
    away_row = _row(managers_json.get("awayManager"), away_team_id) if managers_json else None
    return {"home": home_row, "away": away_row}



def parse_incidents_to_cards_goals(incidents_json, home_team_id: int | None, away_team_id: int | None):
    """
    Returns:
      cards_rows: list of dicts for public.cards
      goals_rows: list of dicts for public.goals
      subs_meta : list of tuples ("on"/"off", player_id, minute, added_time)

    Notes:
    - team_id is inferred from isHome flag (home_team_id / away_team_id).
    - HT/FT 'period' rows (with addedTime=999) are ignored for time calculations.
    - For 'secondYellow' / 'yellowRed' we set yellow=True, yellow_2=True, red=True.
    """
    def _safe_int(x):
        try:
            return int(x) if x is not None else None
        except Exception:
            return None

    def _added_time_norm(x):
        x = _safe_int(x)
        # SofaScore uses 999 as a sentinel on period markers (HT/FT) — ignore for minutes
        return None if (x is None or x >= 900) else x

    def _match_minute(minute, added):
        if minute is None:
            return None
        return minute + (added or 0)

    def _team_from_flag(is_home):
        if is_home is True:
            return home_team_id
        if is_home is False:
            return away_team_id
        return None

    def _half_from_minute(minute):
        # crude but practical; extend if you later ingest extra time
        if minute is None:
            return None
        if minute <= 45:
            return "1"
        if minute <= 90:
            return "2"
        if minute <= 105:
            return "ET1"
        if minute <= 120:
            return "ET2"
        return None

    cards_rows, goals_rows, subs_meta = [], [], []

    # Accept either dict-wrapped {"incidents":[...]} or a plain list
    items = []
    if isinstance(incidents_json, dict):
        items = incidents_json.get("incidents") or []
    elif isinstance(incidents_json, list):
        items = incidents_json
    else:
        return cards_rows, goals_rows, subs_meta

    for inc in items:
        itype = inc.get("incidentType") or inc.get("type")
        iclass = inc.get("incidentClass")  # e.g., "yellow", "red", "regular"
        is_home = inc.get("isHome")
        team_id = _team_from_flag(is_home)

        minute = _safe_int(inc.get("time"))
        added = _added_time_norm(inc.get("addedTime"))
        match_minute = _match_minute(minute, added)
        half = _half_from_minute(minute)

        # --- Substitutions ---
        if itype == "substitution":
            pin = (inc.get("playerIn") or {}).get("id")
            pout = (inc.get("playerOut") or {}).get("id")
            if pin:
                subs_meta.append(("on", _safe_int(pin), minute, added))
            if pout:
                subs_meta.append(("off", _safe_int(pout), minute, added))
            continue  # substitutions do not produce card/goal rows

        # --- Ignore non-sporting markers (period/injuryTime) ---
        if itype in {"period", "injuryTime"}:
            continue

        # --- Cards ---
        if itype == "card" or iclass in {"yellow", "red", "yellowRed", "secondYellow"}:
            player_id = _safe_int((inc.get("player") or {}).get("id"))
            reason = inc.get("reason") or inc.get("text") or None

            yellow = iclass in {"yellow", "yellowRed", "secondYellow"}
            yellow_2 = iclass in {"yellowRed", "secondYellow"}
            red = iclass in {"red", "yellowRed", "secondYellow"}

            cards_rows.append(
                {
                    "fixture_id": None,           # fill in caller
                    "team_id": _safe_int(team_id),
                    "player_id": player_id,
                    "card_minute": minute,
                    "match_minute": match_minute,
                    "yellow": bool(yellow),
                    "yellow_2": bool(yellow_2),   # map to "yellow 2" at insert if your column name has a space
                    "red": bool(red),
                    "reason": reason,
                    "added_time": added,
                }
            )
            continue

        # --- Goals ---
        if itype in {"goal", "ownGoal", "penalty", "missedPenalty"}:
            player_id = _safe_int((inc.get("player") or {}).get("id"))
            is_own = bool(inc.get("isOwnGoal") or inc.get("ownGoal") or (itype == "ownGoal"))

            goals_rows.append(
                {
                    "fixture_id": None,                 # fill in caller
                    "team_id": _safe_int(team_id),
                    "player_id": player_id,
                    "goal_minute": minute,
                    "added_time": added,
                    "match_minute": match_minute,
                    "half": half,
                    "type": itype,                       # keep raw type for transparency
                    "is_own_goal": is_own,
                }
            )
            continue

        # Nothing else to do for other incident types (fouls, corners, etc.)
        # You can extend here later as your schema grows.

    return cards_rows, goals_rows, subs_meta


def parse_shotmap(shotmap_json, home_team_id: int | None, away_team_id: int | None):
    """
    Build rows for your `shots` (or similar) table from SofaScore shotmap payloads.

    Returns:
      shots_rows: list[dict]
    
    Columns produced (fill fixture_id in the caller):
      fixture_id, team_id, player_id,
      minute, added_time, match_minute, half,
      xg, xgot,
      shot_type, goal_type, situation, body_part, outcome_location,
      is_goal, is_on_target, is_blocked, hit_post, is_penalty,
      shooter_x, shooter_y, shooter_z,
      goal_mouth_x, goal_mouth_y, goal_mouth_z,
      block_x, block_y, block_z
    """
    def _safe_int(x):
        try:
            return int(x) if x is not None else None
        except Exception:
            return None

    def _safe_float(x):
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    def _added_time_norm(x):
        # Some feeds omit or give 0; keep None if missing.
        x = _safe_int(x)
        return None if x is None else x

    def _match_minute(minute, added):
        if minute is None:
            return None
        return minute + (added or 0)

    def _team_from_flag(is_home):
        if is_home is True:
            return home_team_id
        if is_home is False:
            return away_team_id
        return None

    def _half_from_minute(minute):
        if minute is None:
            return None
        if minute <= 45:
            return "1"
        if minute <= 90:
            return "2"
        if minute <= 105:
            return "ET1"
        if minute <= 120:
            return "ET2"
        return None

    def _flags_from_type(shot_type: str | None):
        # shotType seen in feed: "goal", "save", "block", "miss", "post"
        st = (shot_type or "").lower()
        is_goal   = st == "goal"
        is_block  = st == "block"
        hit_post  = st == "post"
        # "on target" = goals and saves (posts are off target; blocks depend on definition — keep as blocked only)
        is_on_tgt = st in {"goal", "save"}
        return is_goal, is_on_tgt, is_block, hit_post

    shots_rows: list[dict] = []

    # Accept {"shotmap":[...]} or a plain list
    items = []
    if isinstance(shotmap_json, dict):
        items = shotmap_json.get("shotmap") or []
    elif isinstance(shotmap_json, list):
        items = shotmap_json
    else:
        return shots_rows

    for s in items:
        # Base fields
        player = s.get("player") or {}
        player_id = _safe_int(player.get("id"))
        is_home = s.get("isHome")
        team_id = _team_from_flag(is_home)

        minute = _safe_int(s.get("time"))
        added = _added_time_norm(s.get("addedTime"))
        match_minute = _match_minute(minute, added)
        half = _half_from_minute(minute)

        shot_type = s.get("shotType")          # goal/save/block/miss/post
        goal_type = s.get("goalType")          # regular/penalty/own-goal/etc. (mostly on goals)
        situation = s.get("situation")         # regular/assisted/corner/penalty/fast-break...
        body_part = s.get("bodyPart")          # right-foot/left-foot/head/...
        outcome_location = s.get("goalMouthLocation")  # low-left/high-right/etc.

        is_goal, is_on_target, is_blocked, hit_post = _flags_from_type(shot_type)
        is_penalty = (situation == "penalty") or (goal_type == "penalty")

        xg = _safe_float(s.get("xg"))
        xgot = _safe_float(s.get("xgot"))

        # Coordinates (SofaScore uses % of pitch / goal mouth)
        pcoords = s.get("playerCoordinates") or {}
        gcoords = s.get("goalMouthCoordinates") or {}
        bcoords = s.get("blockCoordinates") or {}

        shooter_x = _safe_float(pcoords.get("x"))
        shooter_y = _safe_float(pcoords.get("y"))
        shooter_z = _safe_float(pcoords.get("z"))

        goal_mouth_x = _safe_float(gcoords.get("x"))
        goal_mouth_y = _safe_float(gcoords.get("y"))
        goal_mouth_z = _safe_float(gcoords.get("z"))

        block_x = _safe_float(bcoords.get("x"))
        block_y = _safe_float(bcoords.get("y"))
        block_z = _safe_float(bcoords.get("z"))

        shots_rows.append(
            {
                "fixture_id": None,               # fill in caller
                "team_id": team_id,
                "player_id": player_id,

                "minute": minute,
                "added_time": added,
                "match_minute": match_minute,
                "half": half,

                "xg": xg,
                "xgot": xgot,

                "shot_type": shot_type,
                "goal_type": goal_type,
                "situation": situation,
                "body_part": body_part,
                "outcome_location": outcome_location,

                "is_goal": bool(is_goal),
                "is_on_target": bool(is_on_target),
                "is_blocked": bool(is_blocked),
                "hit_post": bool(hit_post),
                "is_penalty": bool(is_penalty),

                "shooter_x": shooter_x,
                "shooter_y": shooter_y,
                "shooter_z": shooter_z,

                "goal_mouth_x": goal_mouth_x,
                "goal_mouth_y": goal_mouth_y,
                "goal_mouth_z": goal_mouth_z,

                "block_x": block_x,
                "block_y": block_y,
                "block_z": block_z,
            }
        )

    return shots_rows


# ------------------ UI Shell ------------------
add_common_page_elements()
st.title("🎯 Add Incidents / Lineups / Managers / Shotmap (Single Fixture)")

# Fixture + Team labels
fixtures = get_fixtures()
teams = get_teams()
team_name_by_id = {t["team_id"]: (t.get("name") or str(t["team_id"])) for t in teams}

valid_fixtures = []
for f in fixtures:
    if f.get("id") is None or f.get("fixture_id") is None:
        continue
    home_name = team_name_by_id.get(f.get("home_team_id"), str(f.get("home_team_id")))
    away_name = team_name_by_id.get(f.get("away_team_id"), str(f.get("away_team_id")))
    label = f"{f['fixture_id']} – {home_name} vs {away_name} (R{f.get('round')}, S{f.get('season_id')})"
    valid_fixtures.append({**f, "name": label})

fixture_names = [f["name"] for f in valid_fixtures]
fixture_map = {f["name"]: f for f in valid_fixtures}

selected_fixture_name = st.selectbox("Select Fixture", fixture_names)
fixture = fixture_map[selected_fixture_name]

# ------------------ Action ------------------
if st.button("Fetch & Diff"):
    fixture_id = int(fixture["fixture_id"])
    home_tid = _safe_int(fixture.get("home_team_id"))
    away_tid = _safe_int(fixture.get("away_team_id"))

    with st.spinner("Fetching lineups…"):
        lineups = fetch_lineups(fixture_id)
    with st.spinner("Fetching managers…"):
        managers = fetch_managers(fixture_id)
    with st.spinner("Fetching incidents…"):
        incidents = fetch_incidents(fixture_id)
    with st.spinner("Fetching shotmap…"):
        shotmap = fetch_shotmap(fixture_id)

    # ---- Parse ----
    players_rows, pf_rows = parse_players_from_lineups(lineups, home_tid, away_tid)
    mgr_rows_dict = parse_managers(managers, home_tid, away_tid)
    mgr_list = [r for r in mgr_rows_dict.values() if r]
    card_rows, goal_rows, subs_meta = parse_incidents_to_cards_goals(incidents, home_tid, away_tid)
    shot_rows = parse_shotmap(shotmap, home_tid, away_tid)

    # fill fixture_id
    for r in pf_rows:   r["fixture_id"] = fixture_id
    for r in card_rows: r["fixture_id"] = fixture_id
    for r in goal_rows: r["fixture_id"] = fixture_id
    for r in shot_rows: r["fixture_id"] = fixture_id

    # ---- Diff: players ----
    existing_player_ids = get_existing_players_ids()
    new_players = [r for r in players_rows if r.get("player_id") not in existing_player_ids]

    # ---- Diff: managers ----
    existing_manager_ids = get_existing_managers_ids()
    new_managers = [r for r in mgr_list if r.get("manager_id") and r["manager_id"] not in existing_manager_ids]

    # ---- Diff: players_fixtures ----
    existing_pf_keys = get_existing_pf_for_fixture(fixture_id)
    new_pf = []
    for r in pf_rows:
        key = (r.get("player_id"), r.get("team_id"))
        if key not in existing_pf_keys:
            new_pf.append(r)

    # Enrich PF with subs meta
    subs_on_times, subs_off_times = {}, {}
    for flag, pid, minute, added in subs_meta:
        if not pid:
            continue
        if flag == "on":
            subs_on_times[pid] = {"substituted_on": True, "subbed_on_time": minute}
        else:
            subs_off_times[pid] = {"substituted_off": True, "subbed_off_time": minute}
    for r in new_pf:
        pid = r.get("player_id")
        if pid in subs_on_times:  r.update(subs_on_times[pid])
        if pid in subs_off_times: r.update(subs_off_times[pid])

    # ---- Diff: cards ----
    existing_card_keys = get_existing_cards_for_fixture(fixture_id)
    new_cards = []
    for r in card_rows:
        key = (
            r.get("player_id"),
            r.get("team_id"),
            r.get("card_minute"),
            r.get("added_time"),
            bool(r.get("yellow")),
            bool(r.get("yellow_2")),
            bool(r.get("red")),
        )
        if key not in existing_card_keys:
            new_cards.append(r)

    # ---- Diff: goals ----
    existing_goal_keys = get_existing_goals_for_fixture(fixture_id)
    new_goals = []
    for r in goal_rows:
        key = (
            r.get("player_id"),
            r.get("team_id"),
            r.get("goal_minute"),
            r.get("added_time"),
            r.get("type"),
            bool(r.get("is_own_goal")),
        )
        if key not in existing_goal_keys:
            new_goals.append(r)

    # ---- Diff: shots ----
    have_shot_ids, have_shot_fallback = get_existing_shots_for_fixture(fixture_id)
    new_shots = []
    for r in shot_rows:
        sid = r.get("shot_id")
        if sid is not None:
            if sid not in have_shot_ids:
                new_shots.append(r)
        else:
            key = (r.get("player_id"), r.get("team_id"), r.get("minute"), r.get("added_time"), r.get("situation"))
            if key not in have_shot_fallback:
                new_shots.append(r)

    # ------------------ UI: metrics + tables + inserts ------------------
    st.subheader("✅ Diff Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Players (new)", len(new_players))
        st.metric("Managers (new)", len(new_managers))
    with c2:
        st.metric("Players-Fixtures (new)", len(new_pf))
        st.metric("Cards (new)", len(new_cards))
    with c3:
        st.metric("Goals (new)", len(new_goals))
        st.metric("Shots (new)", len(new_shots))

    def _df_or_empty(rows, cols=None):
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols or [])

    with st.expander("👥 New Players"):
        st.dataframe(_df_or_empty(new_players, ["player_id","name","short_name","team_id"]))
        if new_players and st.button("➕ Insert Players"):
            res = supabase.table("players").insert(new_players).execute()
            st.success(f"Inserted {len(res.data or [])} players")

    with st.expander("🧠 New Managers"):
        st.dataframe(_df_or_empty(new_managers, ["manager_id","name","short_name","slug"]))
        if new_managers and st.button("➕ Insert Managers"):
            res = supabase.table("managers").insert(new_managers).execute()
            st.success(f"Inserted {len(res.data or [])} managers")

    with st.expander("📋 New Players-Fixtures"):
        st.dataframe(_df_or_empty(new_pf, ["player_id","team_id","fixture_id","started","substitute","subbed_on_time","subbed_off_time","minutes_played"]))
        if new_pf and st.button("➕ Insert Players-Fixtures"):
            res = supabase.table("players_fixtures").insert(new_pf).execute()
            st.success(f"Inserted {len(res.data or [])} players_fixtures rows")

    with st.expander("🟨🟥 New Cards"):
        st.dataframe(_df_or_empty(new_cards, ["player_id","team_id","card_minute","added_time","yellow","yellow_2","red","reason"]))
        if new_cards and st.button("➕ Insert Cards"):
            rows = [dict(r) for r in new_cards]
            # If your column name is literally "yellow 2", map it here:
            # for r in rows: r["yellow 2"] = r.pop("yellow_2")
            res = supabase.table("cards").insert(rows).execute()
            st.success(f"Inserted {len(res.data or [])} cards")

    with st.expander("🥅 New Goals"):
        st.dataframe(_df_or_empty(new_goals, ["player_id","team_id","goal_minute","added_time","match_minute","half","type","is_own_goal"]))
        if new_goals and st.button("➕ Insert Goals"):
            res = supabase.table("goals").insert(new_goals).execute()
            st.success(f"Inserted {len(res.data or [])} goals")

    with st.expander("🎯 New Shots"):
        st.dataframe(_df_or_empty(new_shots, ["player_id","team_id","minute","added_time","situation","xg","xgot","shot_id"]))
        if new_shots and st.button("➕ Insert Shots"):
            res = supabase.table("shots").insert(new_shots).execute()
            st.success(f"Inserted {len(res.data or [])} shots")
