"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).
"""

# Library imports
import traceback
import copy

import streamlit as st
import pandas as pd
import numpy as np

from utils.extractors.data_fetcher import fetch_json, fetch_seasons_json, fetch_standing_json, fetch_rounds_json, fetch_round_events
from utils.api.tournaments import TOURNAMENTS
from utils.extractors.data_flatten import get_flattened_standings, get_flattened_round_events
from utils.renders.text_renders import render_goal_list
from utils.renders.graph_renders import render_game_state_gantt

from utils.page_components import (
    add_common_page_elements,
)

# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.header("Web Scrapping", divider=True)
st.text("Where and how I get my data")

st.title("Football Tournament & Season Selector")

# Tournament selection
tournament_names = [t["name"] for t in TOURNAMENTS]
selected_tournament_name = st.selectbox("Select a Tournament", tournament_names, index=0)
selected_tournament = next(t for t in TOURNAMENTS if t["name"] == selected_tournament_name)
# st.text(selected_tournament)
# Fetch seasons
# seasons_url = f"https://api.sofascore.com/api/v1/tournament/{selected_tournament['id']}/seasons"
try:
    seasons_response = fetch_seasons_json(selected_tournament)
    seasons_data = seasons_response.get("seasons", [])
    if not seasons_data:
        st.warning("No seasons found")
        st.stop()
    
    st.subheader("Available Seasons")
    # st.dataframe(pd.json_normalize(seasons_data))

    if seasons_data:
        season_names = [f"{s.get('name')} ({s.get('year')})" for s in seasons_data]
        selected_index = st.selectbox("Select a Season", range(len(season_names)), format_func=lambda i: season_names[i], index=1)
        selected_season = seasons_data[selected_index]
        
        standings_response = fetch_standing_json(selected_tournament, selected_season)
        standing_tables = standings_response.get("standings", [])
        if not standing_tables:
            st.warning("No standings found")
            st.stop()       
        
        if standing_tables:
        
            st.subheader("League Standings (Flattened Data)")
            table_df_formatted = get_flattened_standings(standing_tables)
            st.dataframe(table_df_formatted)
            
            rounds_data = fetch_rounds_json(selected_tournament, selected_season)
            
            if "currentRound" in rounds_data and "rounds" in rounds_data:
                current_round = rounds_data["currentRound"].get("round", 0)
                available_rounds = [r["round"] for r in rounds_data["rounds"] if r["round"] <= current_round]

                selected_round = st.selectbox("Select a Round", available_rounds, index=5)

                st.subheader("Selected Round")
                st.write(f"Selected Round: {selected_round}")
                
                round_events = fetch_round_events(selected_tournament, selected_season, selected_round)
                if round_events:
                    
                    filtered_round_events = get_flattened_round_events(round_events)
                    
                    st.subheader("Flattened Round Events Data")
                    st.dataframe(filtered_round_events,
                                    column_config={
                                        "incidents.home_goals": st.column_config.JsonColumn(
                                            "Home Goal Incidents",
                                                help="JSON strings or objects",
                                                width="large",
                                        ),
                                        "incidents.away_goals": st.column_config.JsonColumn(
                                            "Away Goals Incidents",
                                                help="JSON strings or objects",
                                                width="large",
                                        ),
                                        "gameStates": st.column_config.JsonColumn(
                                            "Game States",
                                                help="JSON strings or objects",
                                                width="large",
                                        ),                                       
                                    }
                                )
                                       
                    st.subheader("Selected Fixture")
                    selected_fixture_label = st.selectbox("Select a fixture", filtered_round_events["match_label"], index=1)
                    selected_fixture = filtered_round_events[filtered_round_events["match_label"] == selected_fixture_label].iloc[0]
                    
                    st.subheader(f"Selected Fixture - {selected_fixture_label}")
                    GameMetaData, TeamResult = st.columns(2)
                    
                    with GameMetaData:                    
                        st.subheader("Game Metadata")
                        st.markdown(f"""
                        **🏆 Tournament:** {selected_fixture['tournament.name']} ({selected_fixture['tournament.category.country.name']})  
                        **🗓️ Season:** {selected_fixture['season.name']} ({selected_fixture['season.year']})  
                        **📅 Round:** {selected_fixture['roundInfo.round']}  
                        **⏰ Kickoff:** {selected_fixture['kickoff']}  
                        **⏱️ Total Time:** {selected_fixture['time.totalTime']} min  
                        **➕ Injury Time:** +{selected_fixture['time.injuryTime1']} (1st), +{selected_fixture['time.injuryTime2']} (2nd)
                        """)
                        
                    with TeamResult:
                        st.subheader("Teams & Result")

                        st.markdown(f"""
                        **🏠 Home Team:** {selected_fixture['homeTeam.name']} (ID: {selected_fixture['homeTeam.id']})  
                        **🚌 Away Team:** {selected_fixture['awayTeam.name']} (ID: {selected_fixture['awayTeam.id']})  

                        **🎯 Final Score:** {selected_fixture['homeScore.display']} - {selected_fixture['awayScore.display']}  
                        **📌 Result:** {selected_fixture['result']}

                        **🟥 Red Cards:**  
                        - Home: {0 if pd.isna(selected_fixture['homeRedCards']) else selected_fixture['homeRedCards']}  
                        - Away: {0 if pd.isna(selected_fixture['awayRedCards']) else selected_fixture['awayRedCards']}  
                        """)                        

                    st.subheader("Goal Incidents")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🏠 Home Goals")
                        st.json(selected_fixture["incidents.home_goals"], expanded=False)
                        render_goal_list(selected_fixture["incidents.home_goals"], selected_fixture["homeTeam.name"])

                    with col2:
                        st.markdown("### 🚌 Away Goals")
                        st.json(selected_fixture["incidents.away_goals"], expanded=False)
                        render_goal_list(selected_fixture["incidents.away_goals"], selected_fixture["awayTeam.name"])
                    
                    st.subheader("Game States")
                    st.json(selected_fixture["segments"], expanded=False)
                    st.json(selected_fixture["gameStates"], expanded=False)
                    
                    render_game_state_gantt(
                        selected_fixture["homeTeam.name"],
                        selected_fixture["awayTeam.name"],
                        selected_fixture["match_label"],
                        selected_fixture["time.totalTime"],
                        selected_fixture["time.injuryTime1"],
                        selected_fixture["time.injuryTime2"],
                        selected_fixture["incidents.home_goals"],
                        selected_fixture["incidents.away_goals"],
                        selected_fixture["segments"]
                    )
                    # st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("No events available for this round.")
            else:
                st.warning("No rounds available for this season.")
            
        else:
            st.warning("No standings available for this season.")
    else:
        st.warning("No seasons found.")
except Exception as e:
    st.error(f"Failed to fetch data: {e}")