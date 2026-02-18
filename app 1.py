import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

APP_TITLE = "Fantasy World Cup Tracker"
DATA_DIR = Path(__file__).parent / "data"
MATCHES_FILE = DATA_DIR / "matches.csv"
PICKS_FILE = DATA_DIR / "picks.csv"
PARTICIPANTS_FILE = DATA_DIR / "participants.csv"

GROUP_STAGE = "Group"
KNOCKOUT_STAGE = "Knockout"

TEAMS = [
    "Algeria",
    "Argentina",
    "Australia",
    "Austria",
    "Belgium",
    "Brazil",
    "Cape Verde",
    "Canada",
    "Colombia",
    "Croatia",
    "Curacao",
    "Ivory Coast",
    "Ecuador",
    "Egypt",
    "England",
    "France",
    "Germany",
    "Ghana",
    "Haiti",
    "Iran",
    "Japan",
    "Jordan",
    "South Korea",
    "Mexico",
    "Morocco",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Panama",
    "Paraguay",
    "Portugal",
    "Qatar",
    "Saudi Arabia",
    "Scotland",
    "Senegal",
    "South Africa",
    "Spain",
    "Switzerland",
    "Tunisia",
    "Uruguay",
    "USA",
    "Uzbekistan",
    "UEFA Path A Winner",
    "UEFA Path B Winner",
    "UEFA Path C Winner",
    "UEFA Path D Winner",
    "Interconf Path 1 Winner",
    "Interconf Path 2 Winner",
]

GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "UEFA Path D Winner"],
    "B": ["Canada", "Qatar", "Switzerland", "UEFA Path A Winner"],
    "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D": ["USA", "Australia", "Paraguay", "UEFA Path C Winner"],
    "E": ["Curacao", "Ivory Coast", "Ecuador", "Germany"],
    "F": ["Japan", "Netherlands", "Tunisia", "UEFA Path B Winner"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "I": ["France", "Norway", "Senegal", "Interconf Path 2 Winner"],
    "J": ["Algeria", "Argentina", "Austria", "Jordan"],
    "K": ["Colombia", "Portugal", "Uzbekistan", "Interconf Path 1 Winner"],
    "L": ["Croatia", "England", "Ghana", "Panama"],
}

GROUP_ORDER = list(GROUPS.keys())
TEAM_TO_GROUP = {
    team: group
    for group, teams in GROUPS.items()
    for team in teams
}



@st.cache_data
def _empty_matches_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "date",
            "stage",
            "team_a",
            "team_b",
            "score_a",
            "score_b",
            "winner",
        ]
    )


@st.cache_data
def _empty_picks_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["participant", "team"])


@st.cache_data
def _empty_participants_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["participant"])


def _load_or_create_csv(path: Path, empty_df: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    empty_df.to_csv(path, index=False)
    return empty_df.copy()


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _bonus_points(goal_diff: int) -> int:
    if goal_diff >= 3:
        return 2
    if goal_diff == 2:
        return 1
    return 0


def _match_points(row: pd.Series) -> dict:
    team_a = str(row["team_a"]).strip()
    team_b = str(row["team_b"]).strip()
    score_a = int(row["score_a"])
    score_b = int(row["score_b"])
    stage = str(row["stage"]).strip()
    winner_override = str(row.get("winner", "")).strip()

    points = {team_a: 0, team_b: 0}

    if score_a == score_b:
        if stage == GROUP_STAGE:
            points[team_a] = 1
            points[team_b] = 1
        elif stage == KNOCKOUT_STAGE and winner_override in {team_a, team_b}:
            points[winner_override] = 2
        return points

    if score_a > score_b:
        winner = team_a
        loser = team_b
    else:
        winner = team_b
        loser = team_a

    goal_diff = abs(score_a - score_b)

    if stage == GROUP_STAGE:
        points[winner] = 3 + _bonus_points(goal_diff)
        points[loser] = 0
    else:
        points[winner] = 2 + _bonus_points(goal_diff)
        points[loser] = 0

    return points


def _normalize_team_name(name: str) -> str:
    return " ".join(str(name).strip().split())


def _clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["team_a"] = df["team_a"].apply(_normalize_team_name)
    df["team_b"] = df["team_b"].apply(_normalize_team_name)
    df["winner"] = df["winner"].fillna("").apply(_normalize_team_name)
    return df


def _resolve_match_winner(row: pd.Series) -> str:
    team_a = str(row.get("team_a", "")).strip()
    team_b = str(row.get("team_b", "")).strip()
    stage = str(row.get("stage", "")).strip()
    winner_override = str(row.get("winner", "")).strip()
    try:
        score_a = int(row.get("score_a", 0))
        score_b = int(row.get("score_b", 0))
    except (TypeError, ValueError):
        return ""

    if score_a == score_b:
        if stage == KNOCKOUT_STAGE and winner_override in {team_a, team_b}:
            return winner_override
        return ""

    return team_a if score_a > score_b else team_b


def _participant_color_map(participants: pd.DataFrame) -> dict:
    palette = [
        "#006847",
        "#FCD116",
        "#0038A8",
        "#D90012",
        "#0B1320",
        "#2E7D32",
        "#F57C00",
        "#1565C0",
        "#C62828",
        "#6A1B9A",
        "#00838F",
        "#AD1457",
    ]
    names = [
        str(name).strip()
        for name in participants.get("participant", []).tolist()
        if str(name).strip()
    ]
    return {name: palette[index % len(palette)] for index, name in enumerate(names)}


def _team_color_map(picks: pd.DataFrame, participant_colors: dict) -> dict:
    team_colors = {}
    for _, row in picks.iterrows():
        participant = str(row.get("participant", "")).strip()
        team = _normalize_team_name(row.get("team", ""))
        if not participant or not team:
            continue
        if team not in team_colors and participant in participant_colors:
            team_colors[team] = participant_colors[participant]
    return team_colors




def _compute_team_points(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=["team", "points"])

    points_map = {}
    for _, row in matches.iterrows():
        if row["team_a"] == "" or row["team_b"] == "":
            continue
        match_points = _match_points(row)
        for team, pts in match_points.items():
            points_map[team] = points_map.get(team, 0) + pts

    points_df = (
        pd.DataFrame(points_map.items(), columns=["team", "points"])
        .sort_values(["points", "team"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return points_df


def _compute_participant_points(points_df: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    if points_df.empty or picks.empty:
        return pd.DataFrame(columns=["participant", "points", "teams"])

    points_lookup = dict(zip(points_df["team"], points_df["points"]))
    summary = []
    for participant, group in picks.groupby("participant"):
        teams = sorted({t for t in group["team"].tolist() if str(t).strip()})
        total = sum(points_lookup.get(team, 0) for team in teams)
        summary.append({"participant": participant, "points": total, "teams": ", ".join(teams)})

    summary_df = pd.DataFrame(summary).sort_values(
        ["points", "participant"], ascending=[False, True]
    )
    return summary_df.reset_index(drop=True)


def _compute_team_points_over_time(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=["date", "team", "points"])

    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    cumulative = {}
    rows = []
    # iterate match-by-match accumulating points
    for _, row in df.iterrows():
        if row["team_a"] == "" or row["team_b"] == "":
            continue
        match_pts = _match_points(row)
        for team, pts in match_pts.items():
            cumulative[team] = cumulative.get(team, 0) + pts
        for team, pts in cumulative.items():
            rows.append({"date": row["date"], "team": team, "points": pts})

    if not rows:
        return pd.DataFrame(columns=["date", "team", "points"])

    out = pd.DataFrame(rows)
    out = out.sort_values(["team", "date"]).reset_index(drop=True)
    return out


def _compute_participant_points_over_time(team_points_time: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    if team_points_time.empty or picks.empty:
        return pd.DataFrame(columns=["date", "participant", "points"])

    # pivot team_points_time so we can sum participant teams per date
    # team_points_time has rows for each match date and team with cumulative points
    # For each date, sum the points for the teams each participant picked
    picks_map = picks.groupby("participant")["team"].apply(list).to_dict()
    rows = []
    for date, group in team_points_time.groupby("date"):
        points_by_team = dict(zip(group["team"], group["points"]))
        for participant, teams in picks_map.items():
            total = sum(points_by_team.get(_normalize_team_name(t), 0) for t in teams)
            rows.append({"date": date, "participant": participant, "points": total})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["participant", "date"]).reset_index(drop=True)
    return out




def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("2026 World Cup format. Points auto-calculated with goal-diff bonus capped at +2.")

    _ensure_data_dir()

    matches = _load_or_create_csv(MATCHES_FILE, _empty_matches_df())
    picks = _load_or_create_csv(PICKS_FILE, _empty_picks_df())
    participants = _load_or_create_csv(PARTICIPANTS_FILE, _empty_participants_df())

    matches = _clean_matches(matches)
    participant_colors = _participant_color_map(participants)
    team_colors = _team_color_map(picks, participant_colors)

    with st.sidebar:
        st.header("Participants")
        with st.form("add_participant"):
            new_participant = st.text_input("Name")
            submit_participant = st.form_submit_button("Add")

        if submit_participant and new_participant.strip():
            name = new_participant.strip()
            if name not in participants["participant"].tolist():
                participants = pd.concat(
                    [participants, pd.DataFrame([{"participant": name}])],
                    ignore_index=True,
                )
                _save_csv(participants, PARTICIPANTS_FILE)
                st.success(f"Added {name}.")
            else:
                st.info("Participant already exists.")

        st.divider()
        st.header("Draft Picks")
        if participants.empty:
            st.info("Add participants to start picking teams.")
        else:
            with st.form("add_pick"):
                participant = st.selectbox(
                    "Participant",
                    options=participants["participant"].tolist(),
                )
                team = st.selectbox(
                    "Team",
                    options=sorted(TEAMS),
                )
                add_pick = st.form_submit_button("Add Pick")

            if add_pick and str(team).strip():
                team_name = _normalize_team_name(team)
                existing_teams = picks.loc[
                    picks["participant"] == participant, "team"
                ].tolist()
                if len(existing_teams) >= 3:
                    st.error("Participants can only select up to 3 teams.")
                elif team_name in existing_teams:
                    st.info("That participant already has this team.")
                else:
                    team_group = TEAM_TO_GROUP.get(team_name)
                    existing_groups = {
                        TEAM_TO_GROUP.get(t)
                        for t in existing_teams
                        if TEAM_TO_GROUP.get(t)
                    }
                    if team_group and team_group in existing_groups:
                        st.error("Participant already has a team from that group.")
                    else:
                        picks = pd.concat(
                            [picks, pd.DataFrame([{"participant": participant, "team": team_name}])],
                            ignore_index=True,
                        )
                        _save_csv(picks, PICKS_FILE)
                        st.success(f"Added {team_name} for {participant}.")

        if not picks.empty:
            st.subheader("Current Picks")
            def _style_pick_row(row: pd.Series) -> list[str]:
                styles = [""] * len(row)
                color = participant_colors.get(str(row.get("participant", "")).strip())
                for index, col in enumerate(row.index):
                    if col == "team" and color:
                        styles[index] = f"color: {color}; font-weight: 600;"
                return styles

            st.dataframe(picks.style.apply(_style_pick_row, axis=1), width="stretch")

    st.header("Add Match Result")
    date = st.date_input("Match Date", value=datetime.today())
    stage = st.selectbox("Stage", options=[GROUP_STAGE, KNOCKOUT_STAGE])
    
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox(
            "Team A",
            options=sorted(TEAMS),
        )
        score_a = st.number_input("Team A Score", min_value=0, step=1)

    with col2:
        if stage == GROUP_STAGE:
            team_a_group = TEAM_TO_GROUP.get(team_a)
            if team_a_group:
                team_b_options = [
                    t
                    for t in sorted(GROUPS[team_a_group])
                    if t != team_a
                ]
            else:
                team_b_options = [t for t in sorted(TEAMS) if t != team_a]
        else:
            team_b_options = [t for t in sorted(TEAMS) if t != team_a]

        team_b = st.selectbox("Team B", options=team_b_options)
        score_b = st.number_input("Team B Score", min_value=0, step=1)

    submit_match = st.button("Add Match")

    if submit_match:
        team_a_name = _normalize_team_name(team_a)
        team_b_name = _normalize_team_name(team_b)

        if not team_a_name or not team_b_name:
            st.error("Both team names are required.")
        elif team_a_name == team_b_name:
            st.error("Teams must be different.")
        elif stage == GROUP_STAGE and date > datetime(2026, 6, 27).date():
            st.error("Group stage matches cannot be after June 27th, 2026.")
        elif stage == KNOCKOUT_STAGE and date < datetime(2026, 6, 28).date():
            st.error("Knockout matches cannot be before June 28th, 2026.")
        elif stage == GROUP_STAGE and TEAM_TO_GROUP.get(team_a_name) != TEAM_TO_GROUP.get(
            team_b_name
        ):
            st.error("Group stage matches must be within the same group.")
        else:
            match_row = {
                "match_id": str(uuid.uuid4())[:8],
                "date": date.strftime("%Y-%m-%d"),
                "stage": stage,
                "team_a": team_a_name,
                "team_b": team_b_name,
                "score_a": int(score_a),
                "score_b": int(score_b),
                "winner": "",
            }
            matches = pd.concat([matches, pd.DataFrame([match_row])], ignore_index=True)
            _save_csv(matches, MATCHES_FILE)
            st.success("Match added.")

    st.divider()
    st.header("Standings")
    team_points = _compute_team_points(matches)
    participant_points = _compute_participant_points(team_points, picks)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Participant Standings")
        if participant_points.empty:
            st.info("No participant points yet.")
        else:
            def _style_participant_standings(row: pd.Series) -> list[str]:
                styles = [""] * len(row)
                color = participant_colors.get(str(row.get("participant", "")).strip())
                for index, col in enumerate(row.index):
                    if col in {"participant", "teams"} and color:
                        styles[index] = f"color: {color}; font-weight: 600;"
                return styles

            st.dataframe(
                participant_points.style.apply(_style_participant_standings, axis=1),
                width="stretch",
            )

    with col2:
        st.subheader("Team Points")
        if team_points.empty:
            st.info("No team points yet.")
        else:
            def _style_team_points_row(row: pd.Series) -> list[str]:
                styles = [""] * len(row)
                color = team_colors.get(str(row.get("team", "")).strip())
                for index, col in enumerate(row.index):
                    if col == "team" and color:
                        styles[index] = f"color: {color}; font-weight: 600;"
                return styles

            st.dataframe(team_points.style.apply(_style_team_points_row, axis=1), width="stretch")


    st.divider()
    st.header("Points Over Time")
    # compute cumulative points over time and participant totals
    team_points_time = _compute_team_points_over_time(matches)
    participant_points_time = _compute_participant_points_over_time(team_points_time, picks)

    if participant_points_time.empty:
        st.info("Not enough data yet to show points over time.")
    else:
        participant_points_time["date"] = pd.to_datetime(participant_points_time["date"])
        # ensure participant color mapping covers participants present in picks
        domains = list(participant_colors.keys())
        ranges = [participant_colors.get(p) for p in domains]

        chart = (
            alt.Chart(participant_points_time)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("points:Q", title="Points"),
                color=alt.Color(
                    "participant:N",
                    scale=alt.Scale(domain=domains, range=ranges),
                    legend=alt.Legend(title="Participant"),
                ),
                tooltip=[alt.Tooltip("date:T", title="Date"), "participant:N", "points:Q"],
            )
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.header("Match Log")
    if matches.empty:
        st.info("No matches recorded yet.")
    else:
        match_log = matches.sort_values("date").drop(columns=["winner"], errors="ignore")

        def _highlight_winner(row: pd.Series) -> list[str]:
            winner = _resolve_match_winner(row)
            styles = []
            for col in row.index:
                cell_value = str(row.get(col, "")).strip()
                cell_color = team_colors.get(cell_value)
                cell_style = ""
                if cell_color and col in {"team_a", "team_b", "winner"}:
                    cell_style += f"color: {cell_color}; font-weight: 600;"
                if col == "team_a" and winner == row["team_a"]:
                    cell_style += "background-color: #c6f6d5;"
                elif col == "team_b" and winner == row["team_b"]:
                    cell_style += "background-color: #c6f6d5;"
                styles.append(cell_style)
            return styles

        st.dataframe(match_log.style.apply(_highlight_winner, axis=1), width="stretch")
        delete_ids = st.multiselect("Delete matches", options=matches["match_id"].tolist())
        if st.button("Delete Selected") and delete_ids:
            matches = matches[~matches["match_id"].isin(delete_ids)].reset_index(drop=True)
            _save_csv(matches, MATCHES_FILE)
            st.success("Deleted selected matches.")


if __name__ == "__main__":
    main()
