import streamlit as st
import pandas as pd
from supabase import create_client
from utils.extractors.data_fetcher import fetch_standing_json
from utils.page_components import add_common_page_elements

# Supabase client
supabase = create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])

# Shared UI elements
add_common_page_elements()
st.title("🏟️ Add Teams from SofaScore Standings")

# Fetch tournaments
@st.cache_data
def get_tournaments():
    res = supabase.table("tournaments").select("id, name, tournament_id, unique_tournament_id").execute()
    return res.data or []

@st.cache_data
def get_seasons(tournament_id):
    res = supabase.table("seasons").select("id, season_id, name, year").eq("tournament_id", tournament_id).execute()
    return res.data or []

def get_existing_teams():
    res = supabase.table("teams").select("team_id").execute()
    return {t["team_id"] for t in res.data} if res.data else set()

def insert_teams(teams):
    return supabase.table("teams").insert(teams).execute()

def fetch_teams_from_standings(standings_json):
    rows = standings_json.get("standings", [])
    teams = []
    for row in rows:
        for item in row.get("rows", []):
            team = item.get("team", {})
            if team:
                teams.append({
                    "team_id": team["id"],
                    "name": team.get("name"),
                    "nameCode": team.get("nameCode"),
                    "teamColours": team.get("teamColors", {}),
                    "crest": f"https://img.sofascore.com/api/v1/team/{team['id']}/image"
                })
    return teams

# --- UI Flow ---

# Tournament selection
tournaments = get_tournaments()
# Build full tournament dicts
valid_tournaments = [
    {
        "name": t["name"],
        "id": t["tournament_id"],
        "unique_tournament": t["unique_tournament_id"],
    }
    for t in tournaments
    if t.get("name") and t.get("tournament_id") is not None and t.get("unique_tournament_id") is not None
]

tournament_names = [t["name"] for t in valid_tournaments]
tournament_map = {t["name"]: t for t in valid_tournaments}

selected_tournament_name = st.selectbox("Select Tournament", tournament_names)
tournament = tournament_map[selected_tournament_name]


# Get seasons from Supabase for the selected tournament
seasons = get_seasons(tournament["id"])

# Match format: {"name": ..., "year": ..., "id": season_id}
valid_seasons = [
    {
        "name": s["name"],
        "year": s["year"],
        "id": s["season_id"]
    }
    for s in seasons
    if s.get("name") and s.get("season_id") is not None
]

season_names = ["All Seasons"] + [s["name"] for s in valid_seasons]
season_map = {s["name"]: s for s in valid_seasons}

selected_season_name = st.selectbox("Select Season", season_names)
# season = season_map[selected_season_name]

# Fetch teams
if st.button("Fetch Teams from SofaScore Standings"):
    selected_seasons = valid_seasons if selected_season_name == "All Seasons" else [season_map[selected_season_name]]

    all_api_teams = []
    for season in selected_seasons:
        standings_json = fetch_standing_json(tournament, season)
        season_teams = fetch_teams_from_standings(standings_json)
        all_api_teams.extend(season_teams)

    if not all_api_teams:
        st.warning("No teams found")
        st.stop()

    # Deduplicate by team_id
    unique_teams = {team["team_id"]: team for team in all_api_teams}.values()

    # Check against existing team_ids
    existing_ids = get_existing_teams()
    already_added = [t for t in unique_teams if t["team_id"] in existing_ids]
    not_added = [t for t in unique_teams if t["team_id"] not in existing_ids]

    # Store in session
    st.session_state["teams_already"] = already_added
    st.session_state["teams_new"] = not_added
    st.session_state["teams_fetched"] = True


# Display results
if st.session_state.get("teams_fetched"):
    st.subheader("✅ Teams Already in Supabase")
    if st.session_state["teams_already"]:
        st.dataframe(pd.DataFrame(st.session_state["teams_already"]))
    else:
        st.info("None yet.")

    st.subheader("🆕 Teams to Add")
    if st.session_state["teams_new"]:
        st.dataframe(pd.DataFrame(st.session_state["teams_new"]))
        if st.button("➕ Add New Teams to Supabase"):
            result = insert_teams(st.session_state["teams_new"])
            if hasattr(result, "data"):
                st.success(f"{len(result.data)} teams inserted")
                st.session_state["teams_fetched"] = False  # reset view
            else:
                st.error("Insert failed.")
    else:
        st.success("All teams already added.")
 