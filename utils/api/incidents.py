import streamlit as st
from utils.extractors.data_fetcher import fetch_json

def extract_goal_incidents(base_row):
    home_team_id = base_row["homeTeam.id"]
    away_team_id = base_row["awayTeam.id"]
    
    home_team_name = base_row["homeTeam.name"]
    away_team_name = base_row["awayTeam.name"]
    
    # st.write(f"Extracting incidents for {home_team_name} vs {away_team_name}")
    
    max_injuryTime1 = base_row["time.injuryTime1"] 
    max_injuryTime2 = base_row["time.injuryTime2"]
    
    eventsId = base_row["id"]
    
    incidents_url = f"https://www.sofascore.com/api/v1/event/{eventsId}/incidents"
    # https://www.sofascore.com/api/v1/event/12436870/incidents
    incidents = fetch_json(incidents_url)
    
    home_goals = []
    away_goals = []                       

    for incident in incidents.get("incidents", []):
        if incident.get("incidentType") != "period" and incident.get("time") == 45 and incident.get("addedTime", None) is not None:
            if incident.get("addedTime", 0) > max_injuryTime1:
                    max_injuryTime1 = incident["addedTime"]
        elif incident.get("incidentType") != "period" and incident.get("time") == 90 and incident.get("addedTime", None) is not None:
            if incident.get("addedTime", 0) > max_injuryTime2:
                    max_injuryTime2 = incident["addedTime"]
        
        if incident.get("incidentType") == "goal":
            goal_event = {     
                "matchMinute": incident.get("time"),                            
                "minute": incident.get("time"),
                "half": "1st" if incident.get("time") <= 45 else "2nd",
                "addedTime": incident.get("addedTime", 0), 
                "playerId": incident.get("player", {}).get("id"),
                "player": incident.get("player", {}).get("name"),
                "playerShortName": incident.get("player", {}).get("shortName"),                                    
                "isOwnGoal": incident.get("incidentClass") == "ownGoal",
                "type": incident.get("incidentClass", "regular"),
            }                               
            
            if incident.get("isHome", False):
                goal_event["teamId"] = home_team_id
                goal_event["teamName"] = home_team_name
                home_goals.append(goal_event)
            else:
                goal_event["teamId"] = away_team_id
                goal_event["teamName"] = away_team_name
                away_goals.append(goal_event)
    # st.write(f"Goals: {home_goals} {away_goals}")    
    
    return max_injuryTime1, max_injuryTime2, home_goals, away_goals


def compute_game_states(home_goals, away_goals, total_time, injury_time_1, injury_time_2):
    def enrich_goal_data(goal, team):
        minute = goal["minute"]
        added = goal.get("addedTime") or 0
        time = minute + added
        return {
            "gameMinute": time,
            "gameAddedTime": added,
            "time": time,
            "team": team,
            "ownGoal": goal.get("isOwnGoal", False),
            "half": goal.get("half", "1st")
        }
    def determine_state(score_home, score_away):
        if score_home == score_away:
            return "drawing", "drawing"
        elif score_home > score_away:
            return "winning", "losing"
        else:
            return "losing", "winning"
        
    # st.write(f"Injury Time 1: {injury_time_1}, Injury Time 2: {injury_time_2}")
    
    for goal in home_goals:
        if goal['half'] == "2nd":
            goal['minute'] += injury_time_1
        # if goal.get('addedTime', 0) > 0:
        #     goal['minute'] += goal['addedTime']
            
    for goal in away_goals:
        if goal['half'] == "2nd":
            goal['minute'] += injury_time_1
        # if goal.get('addedTime', 0) > 0:
        #     goal['minute'] += goal['addedTime']

    # st.write(f"Home goal: {home_goals}")
    # st.write(f"Away goal: {away_goals}")
    
    # Step 1: Merge and sort all goals
    goals = [enrich_goal_data(g, "home") for g in home_goals] + \
            [enrich_goal_data(g, "away") for g in away_goals]
    goals.sort(key=lambda x: x["time"])

    first_half_goals = [g for g in goals if g["half"] == "1st"]
    second_half_goals = [g for g in goals if g["half"] == "2nd"] 
    
    # Step 2: Iterate through game segments
    segments = []
    
    halves_times = {
        "1st": 45 + injury_time_1,
        "2nd": 90 + injury_time_2
    }
    score = {"home": 0, "away": 0}
    prev_time = 0

    for goal1 in first_half_goals:
        # Determine segment duration
        current_time = goal1["time"]
        duration = current_time - prev_time

        home_state, away_state = determine_state(score["home"], score["away"])

        segments.append({
            "start": prev_time,
            "end": current_time,
            "duration": duration,
            "home": home_state,
            "away": away_state,
            "half": goal1['half']
        })

        # Update score
        team = goal1["team"]
        score[team] += 1

        prev_time = current_time
        
    if prev_time < halves_times["1st"]:
        # If the last segment is less than the 1st half duration, adjust it to the end of the 1st half
        duration = halves_times["1st"] - prev_time
        
        home_state, away_state = determine_state(score["home"], score["away"])
        
        segments.append({
            "start": prev_time,
            "end": halves_times["1st"],
            "duration": duration,
            "home": home_state,
            "away": away_state,
            "half": "1st"
        })
        prev_time = halves_times["1st"]
    
    # prev_time = 46  # Start second half from 46 minutes
    
    # Process second half goals
    for goal2 in second_half_goals:
        # st.write(f"Processing goal: {goal}")
        # Determine segment duration
        current_time = goal2["time"]
        duration = current_time - prev_time

        home_state, away_state = determine_state(score["home"], score["away"])

        segments.append({
            "start": prev_time,
            "end": current_time,
            "duration": duration,
            "home": home_state,
            "away": away_state,
            "half": goal2['half']
        })

        # Update score
        team = goal2["team"]
        score[team] += 1
        
        prev_time = current_time

    # Step 3: Add final segment(s) until end of game
    if prev_time < total_time:
        duration = total_time - prev_time
    
        home_state, away_state = determine_state(score["home"], score["away"])
            
        segments.append({
                "start": prev_time,
                "end": total_time,
                "duration": duration,
                "home": home_state,
                "away": away_state,
                "half": "2nd"
            })

    # Step 4: Sum up time in each state
    summary = {"home": {"winning": 0, "drawing": 0, "losing": 0},
               "away": {"winning": 0, "drawing": 0, "losing": 0}}

    for seg in segments:
        summary["home"][seg["home"]] += seg["duration"]
        summary["away"][seg["away"]] += seg["duration"]

    return segments, summary
