import streamlit as st
import pandas as pd
from supabase import create_client
from utils.extractors.data_fetcher import fetch_lineups
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

@st.cache_data
def get_fixtures(season_id):
    res = supabase.table("fixtures").select("id, fixture_id, home_team_id, away_team_id, round, kickoff_date_time").eq("season_id", season_id).execute()
    return res.data or []

def get_existing_players():
    res = supabase.table("players").select("player_id").execute()
    return {p["player_id"] for p in res.data} if res.data else set()

def insert_players(players):
    return supabase.table("players").insert(players).execute()

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

# season_names = ["All Seasons"] + [s["name"] for s in valid_seasons]
season_names = [s["name"] for s in valid_seasons]
season_map = {s["name"]: s for s in valid_seasons}

selected_season_name = st.selectbox("Select Season", season_names)

if st.button("Fetch Fixtures"):
    fixtures = get_fixtures(season_map[selected_season_name]["id"])
    if fixtures:
        df = pd.DataFrame(fixtures)
        columns = ["fixture_id", "home_team_id", "away_team_id", "round", "kickoff_date_time"]
        st.dataframe(df[columns])
    else:
        st.info("No fixtures found for the selected season.")
    