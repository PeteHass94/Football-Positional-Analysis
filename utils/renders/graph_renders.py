import streamlit as st
import pandas as pd
import plotly.graph_objects as go


from utils.api.incidents import compute_game_states

def prepare_gantt_data(segments, team_name, team_type, injury_time_1):
    halftime_boundary = 45 + injury_time_1
    colors = {
        "winning": "green",
        "drawing": "blue",
        "losing": "red"
    }

    data = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        state = seg[team_type]

        data.append({
            "Team": team_name,
            "Start": start,
            "End": end,
            "Duration": end - start,
            "State": state,
            "Color": colors[state],
            "Half": seg["half"],
        })

    return data


def plot_game_state_gantt_split(segments, goal_events, home_team_name, away_team_name, injury_time_1, injury_time_2):
    # Prepare data
    home_data = prepare_gantt_data(segments, home_team_name, "home", injury_time_1)
    away_data = prepare_gantt_data(segments, away_team_name, "away", injury_time_1)
    df = pd.DataFrame(home_data + away_data)

    # Add row label for y-axis (e.g., "Brentford - 1st Half")
    df["Row"] = df["Team"] + " - " + df["Half"] + " Half"
    # Add invisible spacer rows to create padding
    spacers = pd.DataFrame([
        {
            "Team": "Spacer",
            "Row": "Home Scorers",
            "Start": 0,
            "End": 0.1,
            "Duration": 0.1,
            "State": "padding",
            "Half": "spacer"
        },
        {
            "Team": "Spacer",
            "Row": "Away Scorers",
            "Start": 0,
            "End": 0.1,
            "Duration": 0.1,
            "State": "padding",
            "Half": "spacer"
        }
    ])

    df = pd.concat([spacers, df], ignore_index=True)
    
    fig = go.Figure()
    color_map = {
        "winning": "green",
        "drawing": "blue",
        "losing": "red"
    }

    # Track which states have been added to avoid duplicate legend entries
    added_legend = set()

    for _, row in df.iterrows():
        if row["State"] == "padding":
            fig.add_trace(go.Bar(
                x=[row["Duration"]],
                y=[row["Row"]],
                base=row["Start"],
                orientation="h",
                marker=dict(color="rgba(0,0,0,0)"),  # transparent
                showlegend=False,
                hoverinfo="skip",
                opacity=0.0
            ))
            continue
        show_legend = row["State"] not in added_legend
        fig.add_trace(go.Bar(
            x=[row["Duration"]],
            y=[row["Row"]],
            base=row["Start"],
            orientation="h",
            marker=dict(color=color_map[row["State"]]),
            name=row["State"].capitalize(),
            hovertemplate=(
                f"<b>{row['Team']}</b><br>"
                f"{row['State'].capitalize()}<br>"
                f"{row['Start']} → {row['End']} min<br>"
                "<extra></extra>"
            ),
            showlegend=show_legend
        ))
        added_legend.add(row["State"])

    # Set x-axis range
    x_max = 90 + injury_time_2 + 7
    fig.update_layout(
        title="Game State Timeline",
        xaxis_title="Minute of the Game",
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=[       
                "Away Scorers",        
                f"{away_team_name} - 2nd Half",
                f"{away_team_name} - 1st Half",
                f"{home_team_name} - 2nd Half",
                f"{home_team_name} - 1st Half",
                "Home Scorers"
            ]
        ),
        barmode="stack",
        xaxis=dict(type="linear", range=[0, x_max], dtick=5),
        height=350,
        legend_title="Game State",
    )

    # Track number of annotations per minute to stagger them
    annotation_tracker = []
    prev_goal_half = "1st"
    prev_goal_team = ""
    
    for g in goal_events:
        minute = g["minute"] + (g.get("addedTime") or 0)
        team = g["team"]
        player = g.get("playerShortName", g.get("player", "Unknown"))
        is_own_goal = g.get("isOwnGoal", False)
        text_matchMinute = f"{g['matchMinute']}'"
        if g.get('addedTime', 0) > 0:
            text_matchMinute += f"+ {g['addedTime']}"
        text = f"{player} {'(OG)' if is_own_goal else ''} - {text_matchMinute}"

        # Count previous goals within ±15 minutes for staggering
        nearby_count = 0
        nearby_count = sum(1 for m in annotation_tracker if abs(m - minute) <= 10)
        annotation_tracker.append(minute)

       # Positioning
        is_home = team == "home"
        base_y = 0.95 if is_home else 0.1
        y_offset = (0.07 * nearby_count) if (prev_goal_team == g['team']) else 0
        y_pos = base_y + y_offset if is_home else (base_y - y_offset)
        
        
        # Clamp y to keep within visible range
        # y_pos = min(max(y_pos, 0.01), 0.99)

        # Reverse yanchor so arrow points **away from the plot**
        # y_anchor = "bottom" if is_home else "top"
        # x_anchor = "right" if ((nearby_count < 1) and (g['half'] == prev_goal_half)) else "left"
        x_anchor = "center"
        y_anchor = "middle"

        vline_color = "goldenrod" if g["team"] == "home" else "silver"
        
        fig.add_vline(
            x=minute,
            line=dict(color=vline_color, dash="dot"),
            annotation=dict(
                text=text,
                yref="paper",
                y=y_pos,
                xanchor=x_anchor,
                yanchor=y_anchor,  
                showarrow=False,
                font_size=10,
                borderpad=1,
                bgcolor="#1E1E2F",
                opacity=1,
            ),
        )
        prev_goal_half = g.get("half", "1st")
        prev_goal_team = g["team"]

    return fig

def render_game_state_gantt(home_team_name, away_team_name, match_label, total_time, injury_time_1, injury_time_2, home_goals, away_goals, segments):
    for g in home_goals:
        g["team"] = "home"
    for g in away_goals:
        g["team"] = "away"

    all_goals = home_goals + away_goals    

    fig = plot_game_state_gantt_split(
        segments,
        all_goals,
        home_team_name,
        away_team_name,
        injury_time_1,
        injury_time_2
    )

    st.subheader("Game State Timeline")
    st.markdown(f"**Match:** {match_label} &nbsp; • &nbsp; **Total Time:** {total_time} min")
    st.plotly_chart(fig, use_container_width=True)

