import streamlit as st
import pandas as pd
from supabase import create_client
from utils.extractors.data_fetcher import fetch_rounds_json, fetch_round_events
from utils.page_components import add_common_page_elements

from datetime import datetime, timezone

# Supabase client
supabase = create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])

# Shared UI elements
add_common_page_elements()
st.title("🏟️ Add Fixtures from SofaScore Rounds")

# Fetch tournaments
@st.cache_data
def get_tournaments():
    res = supabase.table("tournaments").select("id, name, tournament_id, unique_tournament_id").execute()
    return res.data or []

@st.cache_data
def get_seasons(tournament_id):
    res = supabase.table("seasons").select("id, season_id, name, year").eq("tournament_id", tournament_id).execute()
    return res.data or []

def get_existing_fixtures():
    res = supabase.table("fixtures").select("fixture_id").execute()
    return {f["fixture_id"] for f in res.data} if res.data else set()

def insert_fixtures(fixtures):
    return supabase.table("fixtures").insert(fixtures).execute()

def flatten_fixture_row(row):
    
    try:
        kickoff_dt = datetime.fromtimestamp(row["startTimestamp"], tz=timezone.utc)
    except Exception:
        kickoff_dt = None
    
    return {
        "fixture_id": row["id"],
        "fixture_custom_id": row.get("customId"),
        "home_team_id": row["homeTeam"]["id"],
        "away_team_id": row["awayTeam"]["id"],
        "season_id": row["season_id"],
        "round": row["round_id"],
        "kickoff_date_time": kickoff_dt.isoformat() if kickoff_dt else None,        
        "injury_time_1": row.get("time", {}).get("injuryTime1", 0),
        "injury_time_2": row.get("time", {}).get("injuryTime2", 0),
        "total_time": 90 + row.get("time", {}).get("injuryTime1", 0) + row.get("time", {}).get("injuryTime2", 0),
        "home_score": row.get("homeScore", {}).get("current", 0),
        "away_score": row.get("awayScore", {}).get("current", 0),
        "result": (
            "H" if row["homeScore"]["current"] > row["awayScore"]["current"]
            else "A" if row["homeScore"]["current"] < row["awayScore"]["current"]
            else "D"
        )
    }

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

# Fetch Rounds
if st.button("Fetch All Rounds from SofaScore Seasons"):
    # selected_seasons = valid_seasons if selected_season_name == "All Seasons" else [season_map[selected_season_name]]
    selected_seasons = [season_map[selected_season_name]]
    
    all_api_rounds = []
    for season in selected_seasons:
        rounds_json = fetch_rounds_json(tournament, season)
        if "rounds" in rounds_json:
            for rnd in rounds_json["rounds"]:
                rnd["season_id"] = season["id"]  # Add season_id to each round
                rnd["season_name"] = season["name"]  # Optional: for display/debugging
            all_api_rounds.extend(rounds_json["rounds"])
        else:
            st.warning(f"No rounds found for season {season['name']}")
            st.stop()

    if not all_api_rounds:
        st.warning("No rounds available.")
        st.stop()

    st.subheader("📆 Available Rounds")
    rounds_df = pd.DataFrame(all_api_rounds)
    st.dataframe(rounds_df)
    
    # # Check against existing rounds
    # existing_ids = get_existing_fixtures()
    # already_added = [r for r in all_api_rounds if r["fixture_id"] in existing_ids]
    # not_added = [r for r in all_api_rounds if r["fixture_id"] not in existing_ids]

    # Store in session
    st.session_state["rounds_fetched"] = rounds_df
    # st.session_state["rounds_already"] = already_added
    # st.session_state["rounds_new"] = not_added

# Fetch events for all rounds
if "rounds_fetched" in st.session_state and st.button("Fetch Fixtures for All Rounds"):
    rounds_df = st.session_state["rounds_fetched"]
    all_fixtures = []

    for _, round_row in rounds_df.iterrows():
        round_id = round_row["round"]
        season_id = round_row["season_id"]

        # Prepare season and round dicts
        season_obj = {"id": season_id}
        round_events_json = fetch_round_events(tournament, season_obj, round_id)

        events = round_events_json if isinstance(round_events_json, list) else round_events_json.get("events", [])
        if not events:
            st.info(f"No events for round ID {round_id}")
            continue

        for e in events:
            # Just capture everything for now
            if e.get("time"):
                fixture = {**e, "season_id": season_id, "round_id": round_id} 
                if fixture.get("winnerCode") is None:
                    continue
                else:
                    all_fixtures.append(fixture)

        
    if all_fixtures:
        st.subheader(f"📝 {len(all_fixtures)} Fixtures Fetched")
        st.dataframe(pd.DataFrame(all_fixtures))
        st.session_state["fixtures_fetched"] = all_fixtures
    else:
        st.warning("No fixtures found across all rounds.")
        
    # Display Alread Added and New Fixtures
    
    
    st.subheader("Already Added Fixtures")
    existing_ids = get_existing_fixtures()
    
    st.write(f"Existing IDs: {len(existing_ids)}")
    
    already_added = [f for f in all_fixtures if f["id"] in existing_ids]
    not_added = [f for f in all_fixtures if f["id"] not in existing_ids]
    
    st.session_state["fixtures_fetched"] = all_fixtures
    st.session_state["rounds_already"] = already_added
    st.session_state["rounds_new"] = not_added
    
    
# Display results
if st.session_state.get("fixtures_fetched"):
    st.subheader("✅ Fixtures Already in Supabase")
    if st.session_state["rounds_already"]:
        st.dataframe(pd.DataFrame(st.session_state["rounds_already"]))
    else:
        st.info("None yet.")   
    st.subheader("🆕 Fixtures to Add")
    if st.session_state["rounds_new"]:
        st.dataframe(pd.DataFrame(st.session_state["rounds_new"]))
        # if st.button("➕ Add New Fixtures to Supabase"):               
            
            
        #     result = insert_fixtures(st.session_state["rounds_new"])
        #     if hasattr(result, "data"):
        #         st.success(f"{len(result.data)} fixtures inserted")
        #         st.session_state["fixtures_fetched"] = False  # reset view
        #     else:
        #         st.error("Insert failed.")
    else:
        st.info("No new fixtures to add.")



if "rounds_new" in st.session_state: # and st.button("➕ Add Fixtures to Supabase"):
    
    st.subheader("Fixtures to be Added")       
    
    raw_fixtures = st.session_state["rounds_new"]

    insert_rows = []
    for f in raw_fixtures:
        insert_rows.append(flatten_fixture_row(f))
        
    st.dataframe(pd.DataFrame(insert_rows))
    if insert_rows:
        if st.button("➕ Add Fixtures to Supabase"):
            result = insert_fixtures(insert_rows)
            if hasattr(result, "data"):
                st.success(f"🎉 {len(result.data)} fixtures inserted into Supabase")
            else:
                st.error("❌ Fixture insert failed")
    else:
        st.info("No new fixtures to insert.")
                
