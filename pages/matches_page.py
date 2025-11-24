import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from utils.page_components import add_common_page_elements

add_common_page_elements()


@st.cache_data
def load_matches() -> pd.DataFrame:
    """Load and tidy matches.json from the local data folder."""
    path = Path("data") / "matches.json"
    df = pd.read_json(path)

    # Parse datetime
    df["date_time"] = pd.to_datetime(df["date_time"])

    # Flatten team info
    df["home_team_id"] = df["home_team"].apply(lambda x: x["id"])
    df["home_team_name"] = df["home_team"].apply(lambda x: x["short_name"])
    df["away_team_id"] = df["away_team"].apply(lambda x: x["id"])
    df["away_team_name"] = df["away_team"].apply(lambda x: x["short_name"])

    # Convenience columns
    df["match_label"] = (
        df["home_team_name"] + " vs " + df["away_team_name"]
    )
    df["date"] = df["date_time"].dt.date

    return df


def main():
    st.title("📅 Matches overview")

    df = load_matches()

    # ---- Sidebar filters ----
    st.sidebar.header("Filters")

    # Team filter (home or away)
    all_teams = sorted(
        set(df["home_team_name"].unique()) | set(df["away_team_name"].unique())
    )
    selected_teams = st.sidebar.multiselect(
        "Filter by team (home or away)",
        options=all_teams,
        default=all_teams,
    )

    # Status filter
    all_status = sorted(df["status"].unique())
    selected_status = st.sidebar.multiselect(
        "Filter by status",
        options=all_status,
        default=all_status,
    )

    # Date range filter
    min_date, max_date = df["date"].min(), df["date"].max()
    start_date, end_date = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if not isinstance(start_date, pd.Timestamp):
        # Streamlit returns plain date for date_input when single; make sure we handle tuple
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

    # Apply filters
    mask_team = (
        df["home_team_name"].isin(selected_teams)
        | df["away_team_name"].isin(selected_teams)
    )
    mask_status = df["status"].isin(selected_status)
    mask_date = (df["date"] >= start_date.date()) & (df["date"] <= end_date.date())

    df_filt = df[mask_team & mask_status & mask_date].copy()

    # ---- Top summary metrics ----
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matches", len(df_filt))
    with col2:
        st.metric("Unique teams", df_filt[["home_team_name", "away_team_name"]]
                  .stack()
                  .nunique())
    with col3:
        st.metric("Competitions", df_filt["competition_id"].nunique())

    st.markdown("---")

    # ---- Matches table ----
    st.subheader("Match list")

    display_cols = [
        "id",
        "date_time",
        "home_team_name",
        "away_team_name",
        "status",
        "competition_id",
        "season_id",
    ]
    st.dataframe(
        df_filt[display_cols].sort_values("date_time"),
        use_container_width=True,
        hide_index=True,
    )

    # ---- Visual 1: Matches per team ----
    st.subheader("Matches per team")

    # Build a long-format table where each row is team–match
    home = df_filt[["id", "date", "home_team_name"]].rename(
        columns={"home_team_name": "team"}
    )
    away = df_filt[["id", "date", "away_team_name"]].rename(
        columns={"away_team_name": "team"}
    )
    team_matches = pd.concat([home, away], ignore_index=True)

    team_counts = (
        team_matches.groupby("team")["id"]
        .nunique()
        .reset_index()
        .rename(columns={"id": "matches"})
        .sort_values("matches", ascending=False)
    )

    fig_team = px.bar(
        team_counts,
        x="team",
        y="matches",
        title="Number of matches per team (home or away)",
    )
    fig_team.update_layout(xaxis_title="", yaxis_title="Matches")
    st.plotly_chart(fig_team, use_container_width=True)

    # ---- Visual 2: Matches over time ----
    st.subheader("Matches over time")

    matches_per_day = (
        df_filt.groupby("date")["id"].nunique().reset_index().rename(
            columns={"id": "matches"}
        )
    )

    fig_time = px.bar(
        matches_per_day,
        x="date",
        y="matches",
        title="Matches per day",
    )
    fig_time.update_layout(xaxis_title="Date", yaxis_title="Matches")
    st.plotly_chart(fig_time, use_container_width=True)

    # ---- Optional: match selector for detailed view ----
    st.subheader("Match details")

    if not df_filt.empty:
        match_ids = df_filt.sort_values("date_time")["id"].tolist()
        selected_match_id = st.selectbox(
            "Select a match", options=match_ids, format_func=lambda mid: 
            df_filt.loc[df_filt["id"] == mid, "match_label"].iloc[0]
            + " — "
            + df_filt.loc[df_filt["id"] == mid, "date_time"].iloc[0].strftime("%Y-%m-%d %H:%M")
        )

        match_row = df_filt[df_filt["id"] == selected_match_id].iloc[0]

        st.write(
            f"**{match_row.home_team_name} vs {match_row.away_team_name}**  "
            f"on {match_row.date_time.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        st.json(
            {
                "match_id": int(match_row.id),
                "status": match_row.status,
                "competition_id": int(match_row.competition_id),
                "season_id": int(match_row.season_id),
                "competition_edition_id": int(match_row.competition_edition_id),
                "home_team": {
                    "id": int(match_row.home_team_id),
                    "name": match_row.home_team_name,
                },
                "away_team": {
                    "id": int(match_row.away_team_id),
                    "name": match_row.away_team_name,
                },
            }
        )
    else:
        st.info("No matches match the current filter settings.")


if __name__ == "__main__":
    main()
