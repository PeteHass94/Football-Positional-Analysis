import streamlit as st
from supabase import create_client
import pandas as pd

from utils.extractors.data_fetcher import fetch_seasons_json

from utils.page_components import (
    add_common_page_elements,
)

# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

# Setup Supabase
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Get tournaments
@st.cache_data
def get_tournaments():
    response = supabase.table("tournaments").select("*").execute()
    
    # Check if data exists
    if not response.data:
        st.warning("No tournament data returned.")
        st.stop()
    
    st.write("Raw tournament response:")
    st.json(response.data, expanded=False)  # Debug print
    return response.data or []

# Get existing seasons
def get_existing_seasons(tournament_id):
    response = supabase.table("seasons") \
        .select("season_id, name") \
        .eq("tournament_id", tournament_id).execute()
    return response.data if response.data else []

# Insert new seasons
def insert_new_seasons(tournament_id, unique_tournament_id, seasons):
    rows = [
        {
            "season_id": s["season_id"],
            "name": s["name"],
            "year": s["year"],
            "tournament_id": tournament_id,
            "unique_tournament_id": unique_tournament_id
        }
        for s in seasons
    ]
    return supabase.table("seasons").insert(rows).execute()

# UI
st.title("📅 Add Seasons from API")

# Fetch tournaments
tournaments = get_tournaments()

# Handle empty/missing data
if not tournaments:
    st.warning("No tournaments available.")
    st.stop()

# Build list of full tournament dicts for selection
valid_tournaments = [
    {
        "name": t["name"],
        "id": t["tournament_id"],
        "unique_tournament": t["unique_tournament_id"],
    }
    for t in tournaments
    if t.get("name") and t.get("tournament_id") is not None and t.get("unique_tournament_id") is not None
]

# Create selectbox labels from names, store mapping of name -> full object
tournament_labels = [t["name"] for t in valid_tournaments]
tournament_map = {t["name"]: t for t in valid_tournaments}

# Select from dropdown
selected_name = st.selectbox("**Select Tournament**", tournament_labels)
tournament = tournament_map[selected_name]

# Now tournament = {"name": "Premier League", "id": 1, "unique_tournament": 17}
st.write("Selected Tournament:", tournament)


if st.button("Fetch Seasons from API"):
    seasons_response = fetch_seasons_json(tournament)
    api_seasons = seasons_response.get("seasons", [])

    if not api_seasons:
        st.warning("No seasons found")
        st.stop()

    for s in api_seasons:
        s["season_id"] = s["id"]
        s["unique_tournament_id"] = tournament["unique_tournament"]

    existing_seasons = get_existing_seasons(tournament["id"])
    existing_ids = {s["season_id"] for s in existing_seasons}

    already_added = [s for s in api_seasons if s["season_id"] in existing_ids]
    not_added = [s for s in api_seasons if s["season_id"] not in existing_ids]

    # Store all in session_state
    st.session_state["already_added"] = already_added
    st.session_state["new_seasons"] = not_added
    st.session_state["fetched"] = True

if st.session_state.get("fetched"):
    st.subheader("✅ Already Added Seasons")
    already_added = st.session_state.get("already_added", [])
    if already_added:
        st.dataframe(pd.DataFrame(already_added))
    else:
        st.info("No matching seasons found in Supabase")

    st.subheader("🆕 New Seasons to Add")
    new_seasons = st.session_state.get("new_seasons", [])
    if new_seasons:
        st.dataframe(pd.DataFrame(new_seasons))
        if st.button("➕ Add New Seasons to Supabase"):
            insert_response = insert_new_seasons(tournament["id"], tournament["unique_tournament"], new_seasons)

            if hasattr(insert_response, "data"):
                st.success(f"🎉 Added {len(new_seasons)} new seasons")
                st.write(insert_response.data)
                # Optional: clear or refresh
                del st.session_state["fetched"]
                del st.session_state["new_seasons"]
                del st.session_state["already_added"]
            else:
                st.error("Something went wrong inserting new seasons.")
    else:
        st.success("All seasons already added ✅")


if st.button("🔄 Reset"):
    for key in ["fetched", "new_seasons", "already_added"]:
        st.session_state.pop(key, None)
