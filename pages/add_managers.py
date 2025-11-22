# add_managers.py
import streamlit as st
import pandas as pd
from supabase import create_client
from utils.page_components import add_common_page_elements
from utils.extractors.data_fetcher import fetch_managers

# ================== CONFIG (edit these if your column names differ) ==================
FIXTURE_HOME_MANAGER_COL = "home_manager_id"  # e.g. "home_manager_id"
FIXTURE_AWAY_MANAGER_COL = "away_manager_id"  # e.g. "away_manager_id"

# ================== SUPABASE ==================
supabase = create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])

# ================== HELPERS ==================
@st.cache_data(show_spinner=False)
def get_fixtures():
    """
    Expect fixtures table to hold: id (PK), fixture_id (external), home_team_id, away_team_id, season_id, round
    """
    res = supabase.table("fixtures").select(
        "id, fixture_id, home_team_id, away_team_id, season_id, round"
    ).order("fixture_id").execute()
    return res.data or []

@st.cache_data(show_spinner=False)
def get_fixtures_missing_managers():
    """
    Pull fixtures where either home_manager_id OR away_manager_id is NULL.
    """
    res = (
        supabase
        .table("fixtures")
        .select("id, fixture_id, home_team_id, away_team_id, season_id, round")
        .or_(f"{FIXTURE_HOME_MANAGER_COL}.is.null,{FIXTURE_AWAY_MANAGER_COL}.is.null")
        .order("fixture_id")
        .execute()
    )
    return res.data or []

@st.cache_data(show_spinner=False)
def get_teams():
    res = supabase.table("teams").select("team_id, name").execute()
    return res.data or []

def _safe_int(x):
    try:
        return int(x) if x is not None else None
    except Exception:
        return None

def parse_managers_payload(payload: dict, home_team_id: int | None, away_team_id: int | None):
    """
    managers_json example:
    {
      "homeManager": {"name":"Fabian Hurzeler","slug":"fabian-hurzeler","shortName":"F. Hurzeler","id":788529,...},
      "awayManager": {"name":"Thomas Frank","slug":"thomas-frank","shortName":"T. Frank","id":94249,...}
    }
    Returns {"home": row_or_None, "away": row_or_None}
    """
    def _row(mgr: dict | None, team_id: int | None):
        if not mgr:
            return None
        return {
            "name": mgr.get("name"),
            "short_name": mgr.get("shortName"),
            "manager_id": _safe_int(mgr.get("id")),   # external ID from feed
            "slug": mgr.get("slug"),
            "team_id": _safe_int(team_id),
        }
    if not isinstance(payload, dict):
        return {"home": None, "away": None}

    return {
        "home": _row(payload.get("homeManager"), home_team_id),
        "away": _row(payload.get("awayManager"), away_team_id),
    }

def upsert_managers_and_link_fixture(fixture_id: int, rows_dict: dict):
    """
    - Upserts managers on (manager_id, team_id) uniqueness
    - SELECTs back PKs (managers.id) using an OR filter
    - Updates fixtures.{home_manager_id, away_manager_id}
    """
    rows = [r for r in rows_dict.values() if r and r.get("manager_id") and r.get("team_id")]
    if not rows:
        return {"status": "no-managers"}

    # 1) Upsert (no .select() chaining in supabase-py)
    supabase.table("managers").upsert(
        rows,
        on_conflict="manager_id,team_id",
        ignore_duplicates=False
    ).execute()

    # 2) Read back PKs for the (manager_id, team_id) pairs we just upserted
    pairs = [(r["manager_id"], r["team_id"]) for r in rows]
    or_filter = ",".join([f"and(manager_id.eq.{mid},team_id.eq.{tid})" for mid, tid in pairs])

    sel = (
        supabase.table("managers")
        .select("id, manager_id, team_id, name")
        .or_(or_filter)
        .execute()
    )

    returned = sel.data or []
    idx = {(r["manager_id"], r["team_id"]): r["id"] for r in returned if r.get("id") is not None}

    # 3) Map home/away PKs
    home_doc = rows_dict.get("home")
    away_doc = rows_dict.get("away")
    home_pk = idx.get((home_doc["manager_id"], home_doc["team_id"])) if home_doc else None
    away_pk = idx.get((away_doc["manager_id"], away_doc["team_id"])) if away_doc else None

    # 4) Update the fixture row
    update_doc = {}
    if home_pk is not None:
        update_doc[FIXTURE_HOME_MANAGER_COL] = home_pk
    if away_pk is not None:
        update_doc[FIXTURE_AWAY_MANAGER_COL] = away_pk

    if update_doc:
        supabase.table("fixtures").update(update_doc).eq("fixture_id", fixture_id).execute()

    return {
        "status": "linked",
        "home_manager_id": home_pk,
        "away_manager_id": away_pk,
        "returned_count": len(returned),
    }


# ================== UI SHELL ==================
add_common_page_elements()
st.title("🧠 Add Managers → Fixtures")

fixtures = get_fixtures()
teams = get_teams()
team_name_by_id = {t["team_id"]: (t.get("name") or str(t["team_id"])) for t in teams}

if not fixtures:
    st.info("No fixtures found.")
    st.stop()

# ---------- Single-fixture flow ----------
st.subheader("Single fixture")
options = []
for f in fixtures:
    home_name = team_name_by_id.get(f.get("home_team_id"), str(f.get("home_team_id")))
    away_name = team_name_by_id.get(f.get("away_team_id"), str(f.get("away_team_id")))
    label = f'{f["fixture_id"]} – {home_name} vs {away_name} (R{f.get("round")}, S{f.get("season_id")})'
    options.append({**f, "label": label})

choice = st.selectbox("Select a fixture", [o["label"] for o in options])
fixture = next(o for o in options if o["label"] == choice)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.write(f'**Fixture ID:** {fixture["fixture_id"]}')
with col_b:
    st.write(f'**Home team:** {team_name_by_id.get(fixture["home_team_id"], fixture["home_team_id"])}')
with col_c:
    st.write(f'**Away team:** {team_name_by_id.get(fixture["away_team_id"], fixture["away_team_id"])}')

if st.button("Fetch managers and link"):
    fid = int(fixture["fixture_id"])
    home_tid = _safe_int(fixture.get("home_team_id"))
    away_tid = _safe_int(fixture.get("away_team_id"))

    with st.spinner("Fetching managers…"):
        payload = fetch_managers(fid)

    rows_dict = parse_managers_payload(payload, home_tid, away_tid)

    # Preview before upsert
    preview_df = pd.DataFrame([r for r in rows_dict.values() if r])
    with st.expander("Preview parsed managers"):
        st.dataframe(preview_df if not preview_df.empty else pd.DataFrame(columns=["name","short_name","manager_id","slug","team_id"]))

    if not any(rows_dict.values()):
        st.warning("No managers found in payload.")
    else:
        with st.spinner("Upserting managers and updating fixture…"):
            result = upsert_managers_and_link_fixture(fid, rows_dict)
        if result.get("status") == "linked":
            st.success(
                f"Linked fixture {fid} — "
                f'{FIXTURE_HOME_MANAGER_COL}: {result.get("home_manager_id")}, '
                f'{FIXTURE_AWAY_MANAGER_COL}: {result.get("away_manager_id")} '
                f'(matched {result.get("returned_count")} manager rows).'
            )
        else:
            st.info("Nothing to do — no managers payload.")


# ---------- Bulk flow ----------
st.markdown("---")
st.subheader("Bulk process: fixtures missing managers")

missing = get_fixtures_missing_managers()
st.caption(f"{len(missing)} fixtures currently have at least one NULL manager link.")

if missing:
    # Show a compact preview table
    prev_cols = ["fixture_id", "home_team_id", "away_team_id", "season_id", "round"]
    st.dataframe(pd.DataFrame([{k: m.get(k) for k in prev_cols} for m in missing]), use_container_width=True)

    if st.button("Process all fixtures missing managers"):
        results = []
        prog = st.progress(0)
        total = len(missing)

        for i, f in enumerate(missing, start=1):
            fid = _safe_int(f.get("fixture_id"))
            home_tid = _safe_int(f.get("home_team_id"))
            away_tid = _safe_int(f.get("away_team_id"))

            status = "skipped"
            detail = ""
            home_pk = None
            away_pk = None

            try:
                # Skip if no team ids (unlikely, but defensive)
                if not fid or not home_tid or not away_tid:
                    status = "skipped"
                    detail = "missing team_id or fixture_id"
                else:
                    payload = fetch_managers(fid)
                    rows_dict = parse_managers_payload(payload, home_tid, away_tid)

                    if not any(rows_dict.values()):
                        status = "no-managers"
                        detail = "payload empty"
                    else:
                        res = upsert_managers_and_link_fixture(fid, rows_dict)
                        status = res.get("status", "unknown")
                        home_pk = res.get("home_manager_id")
                        away_pk = res.get("away_manager_id")
                        detail = f"matched {res.get('returned_count', 0)} rows"

            except Exception as e:
                status = "error"
                detail = str(e)

            results.append(
                {
                    "fixture_id": fid,
                    "home_team_id": home_tid,
                    "away_team_id": away_tid,
                    "status": status,
                    "home_manager_id": home_pk,
                    "away_manager_id": away_pk,
                    "detail": detail,
                }
            )

            prog.progress(i / max(total, 1))

        st.success("Bulk processing complete.")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
else:
    st.info("All fixtures already have both manager links populated. 🎉")


# ================== Fixtures + Teams + Managers Overview ==================
@st.cache_data(show_spinner=False)
def get_managers_index():
    """
    Returns:
      - managers_by_pk: { managers.id (PK) -> {"name", "manager_id", "team_id"} }
    """
    res = supabase.table("managers").select(
        "id, manager_id, team_id, name"
    ).execute()
    rows = res.data or []
    by_pk = {}
    for r in rows:
        pk = r.get("id")
        if pk is not None:
            by_pk[pk] = {
                "name": r.get("name"),
                "manager_id": r.get("manager_id"),
                "team_id": r.get("team_id"),
            }
    return by_pk

@st.cache_data(show_spinner=False)
def get_fixtures_for_overview():
    # Pull all the columns you want to display
    res = supabase.table("fixtures").select(
        "fixture_id, season_id, round, "
        "home_team_id, away_team_id, "
        f"{FIXTURE_HOME_MANAGER_COL}, {FIXTURE_AWAY_MANAGER_COL}"
    ).order("fixture_id").execute()
    return res.data or []

st.subheader("📑 Fixtures — teams & managers")

# Optional filter: only show fixtures still missing a manager link
show_only_unlinked = st.checkbox("Show only fixtures missing a home/away manager link", value=False)

fixtures_rows = get_fixtures_for_overview()
teams_rows = get_teams()
teams_by_id = {t["team_id"]: (t.get("name") or str(t["team_id"])) for t in teams_rows}
managers_by_pk = get_managers_index()

def _fmt_team(team_id):
    return f"{team_id} — {teams_by_id.get(team_id, 'Unknown Team')}"

def _fmt_manager(pk_id):
    """pk_id is fixtures.home_manager_id / away_manager_id (FK to managers.id)."""
    if not pk_id:
        return None
    m = managers_by_pk.get(pk_id)
    if not m:
        return f"{pk_id} — (missing in managers)"
    # Show: <managers.id PK> — <name> [manager_id=<external> team_id=<team_id>]
    return f"{pk_id} — {m.get('name') or ''} [manager_id={m.get('manager_id')}, team_id={m.get('team_id')}]"

table_rows = []
for f in fixtures_rows:
    h_pk = f.get(FIXTURE_HOME_MANAGER_COL)  # managers.id (PK)
    a_pk = f.get(FIXTURE_AWAY_MANAGER_COL)
    if show_only_unlinked and h_pk and a_pk:
        continue

    table_rows.append({
        "fixture_id": f.get("fixture_id"),
        "season_id": f.get("season_id"),
        "round": f.get("round"),

        "home_team": _fmt_team(f.get("home_team_id")),
        "home_manager": _fmt_manager(h_pk),

        "away_team": _fmt_team(f.get("away_team_id")),
        "away_manager": _fmt_manager(a_pk),
    })

df = pd.DataFrame(table_rows, columns=[
    "fixture_id","season_id","round","home_team","home_manager","away_team","away_manager"
])

st.dataframe(df, use_container_width=True)

# # Optional: export button
# if not df.empty:
#     csv = df.to_csv(index=False).encode("utf-8")
#     st.download_button("⬇️ Download CSV", data=csv, file_name="fixtures_teams_managers.csv", mime="text/csv")
