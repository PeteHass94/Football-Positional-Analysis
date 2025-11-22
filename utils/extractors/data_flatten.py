import streamlit as st
import pandas as pd
from datetime import datetime
from utils.extractors.data_fetcher import fetch_json
from utils.api.incidents import extract_goal_incidents, compute_game_states

# Extract and reshape data
def flatten_table_row(row):
    base = {
        "id": row.get("team", {}).get("id"),
        "team_name": row.get("team", {}).get("name"),
        "position": row.get("position"),
        "wins": row.get("wins"),
        "draws": row.get("draws"),
        "losses": row.get("losses"),
        "scoresFor": row.get("scoresFor"),
        "scoresAgainst": row.get("scoresAgainst"),
        "scoreDiffFormatted": row.get("scoreDiffFormatted"),
        "description": " | ".join(desc["text"] for desc in row.get("descriptions", [])) if row.get("descriptions") else "",
        "team_nameCode": row.get("team", {}).get("nameCode"),
        "league_Result": row.get("promotion", {}).get("text"),
        "team_Colours": row.get("team", {}).get("teamColors"),
        "team_National": row.get("team", {}).get("national")
    }
    # Include all other fields too
    return base


def get_rows(standing_tables):
    return standing_tables[0].get("rows", [])

def get_flattened_standings(standing_tables):
    """
    Extracts and flattens the standings data from the provided standing tables.
    
    Args:
        standing_tables (list): List of standing tables containing team data.
        
    Returns:
        list: List of flattened team data dictionaries.
    """
    rows = get_rows(standing_tables)
    table_data = [flatten_table_row(row) for row in rows]
    table_df = pd.DataFrame(table_data)
    
    # Reorder columns
    table_first_cols = ["id", "team_name", "position", "wins", "draws", "losses", "scoresFor", "scoresAgainst", "scoreDiffFormatted", "description", "team_nameCode", "league_Result", "team_Colours", "team_National"]
    # remaining_cols = [col for col in df.columns if col not in first_cols]
    table_df_formatted = table_df[table_first_cols] #+ remaining_cols]
    table_df_formatted = table_df_formatted.reset_index(drop=True)
    
    table_df_formatted.index = table_df_formatted.index + 1  # Start index at 1
    
    return table_df_formatted


def flatten_round_row(row):
    base = {
        "id": row.get("id"),
        "customId": row.get("customId"),
        
        # Season Details
        "season.name": row.get("season", {}).get("name"),
        "season.year": row.get("season", {}).get("year"),
        "season.id": row.get("season", {}).get("id"),
        "roundInfo.round": row.get("roundInfo", {}).get("round"),
        
        "winnerCode": row.get("winnerCode"),
        "hasGlobalHighlights": row.get("hasGlobalHighlights"),
        "hasXg": row.get("hasXg"),
        "hasEventPlayerStatistics": row.get("hasEventPlayerStatistics"),
        "hasEventPlayerHeatMap": row.get("hasEventPlayerHeatMap"),
        "detailId": row.get("detailId"),                            
        "homeRedCards": row.get("homeRedCards", None),  # Defaults to None if missing
        "awayRedCards": row.get("awayRedCards", None),  # Defaults to None if missing
        "slug": row.get("slug"),
        "startTimestamp": row.get("startTimestamp"),
        
        # Tournament Details
        "tournament.name": row.get("tournament", {}).get("name"),
        "tournament.slug": row.get("tournament", {}).get("slug"),
        "tournament.category.country.name": row.get("tournament", {}).get("category", {}).get("country", {}).get("name"),
        
        # Teams Details
        "homeTeam.id": row.get("homeTeam", {}).get("id"),
        "homeTeam.name": row.get("homeTeam", {}).get("name"),
        "homeTeam.slug": row.get("homeTeam", {}).get("slug"),
        "awayTeam.id": row.get("homeTeam", {}).get("id"),
        "awayTeam.name": row.get("awayTeam", {}).get("name"),
        "awayTeam.slug": row.get("awayTeam", {}).get("slug"),                        
        
        # Scores
        "homeScore.display": row.get("homeScore", {}).get("display"),
        "awayScore.display": row.get("awayScore", {}).get("display"),
        
        
        # Times
        "time.injuryTime1": row.get("time", {}).get("injuryTime1"),
        "time.injuryTime2": row.get("time", {}).get("injuryTime2"),
    }
    
    base["match_label"] = f"{base['homeTeam.name']} vs {base['awayTeam.name']} - {base['season.name']}"
    base["result"] = "Home" if base["homeScore.display"] > base["awayScore.display"] else "Away" if base["homeScore.display"] < base["awayScore.display"] else "Draw"          
    base["kickoff"] = datetime.fromtimestamp(base["startTimestamp"]).strftime("%Y-%m-%d %H:%M")

    base["time.injuryTime1"], base["time.injuryTime2"], base["incidents.home_goals"], base["incidents.away_goals"] = extract_goal_incidents(base)
    
    # st.json(base["incidents.home_goals"], expanded=False)
    # st.write(f"base {base['incidents.home_goals']} {base['incidents.away_goals']}")
    
    base["time.totalTime"] = 90 + base["time.injuryTime1"] + base["time.injuryTime2"]
    
    base["segments"], base["gameStates"] = compute_game_states(
        base["incidents.home_goals"],
        base["incidents.away_goals"],
        base["time.totalTime"],
        base["time.injuryTime1"],
        base["time.injuryTime2"]
    )
    
    # Include any other columns you want in the same format:
    # "column_name": row.get("column_name", default_value)
    
    return base

def get_flattened_round_events(round_events):
    """
    Extracts and flattened the round events data from the provided round events.
    
    Args:
        round_events (list): List of round events containing match data.
        
    Returns:
        list: List of flattened match data dictionaries.
    """
    flattened_data = [flatten_round_row(row) for row in round_events]
    return pd.DataFrame(flattened_data)
    # df = pd.DataFrame(flattened_data)   
    
    # # Reorder columns to put match_label first
    # if 'match_label' in df.columns:
    #     first_cols = ['match_label']
    #     remaining_cols = [col for col in df.columns if col not in first_cols]
    #     df = df[first_cols + remaining_cols]
    
    
    