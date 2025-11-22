import streamlit as st
import datetime as dt
import json
from supabase import create_client
import requests

st.set_page_config(page_title="Admin • Monthly Quiz", page_icon="🛠️")

# ---- Secrets & clients ----
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")  # optional if you swap provider
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---- Helpers ----
def current_user():
    # Streamlit doesn't do auth itself; use Supabase auth via js widget or magic link flow you already set up
    # For admin gating in Streamlit, store user_id in session_state after login
    return st.session_state.get("user_id")

def is_admin(user_id):
    if not user_id: return False
    res = supabase.table("admins").select("*").eq("user_id", user_id).execute()
    return len(res.data) == 1

def month_start(year, month):
    return dt.date(year, month, 1)

def call_llm(news_item, month_label):
    if not OPENAI_API_KEY:
        st.warning("Set OPENAI_API_KEY in secrets to enable AI suggestions.")
        return None
    prompt = f"""
You are generating a multiple-choice question about last month's news.
Given one news item below, produce a JSON object with fields:
- prompt
- options (array of 4 strings)
- correct_index (0-3)
- hint
- explanation
- source_url

Constraints:
- Avoid ambiguity; ensure only one correct answer.
- Keep it concise (question 1 sentence).
- For Northern Ireland items, keep regional context explicit.

NEWS_ITEM:
Title: {news_item['title']}
Summary: {news_item['summary']}
Source: {news_item['url']}
Month: {month_label}

Return ONLY JSON.
"""
    # OpenAI SDK (chat)
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        st.error("AI returned non-JSON. You can copy/paste and fix below.")
        st.code(content)
        return None

def upsert_quiz(month_date, category_id, title):
    data = {"month": str(month_date), "category": category_id, "title": title}
    res = supabase.table("quizzes").upsert(data, on_conflict="month,category").select("*").execute()
    return res.data[0]

def insert_q_and_choices(quiz_id, draft):
    q = supabase.table("questions").insert({
        "quiz_id": quiz_id,
        "prompt": draft["prompt"],
        "explanation": draft.get("explanation",""),
        "source_url": draft.get("source_url","")
    }).select("*").execute().data[0]
    for i,opt in enumerate(draft["options"]):
        supabase.table("choices").insert({
            "question_id": q["id"],
            "choice_text": opt,
            "is_correct": i == draft["correct_index"]
        }).execute()
    return q

# ---- UI ----
st.title("🛠️ Admin: Monthly Quiz Builder")

uid = current_user()
if not is_admin(uid):
    st.error("Admins only. Please sign in and ensure your account is added to the admins table.")
    st.stop()

# Category pick
cats = supabase.table("categories").select("id,slug,name").order("name").execute().data
cat_name_to_id = {c["name"]: c["id"] for c in cats}
cat = st.selectbox("Category", [c["name"] for c in cats])

# Month/year
col1, col2 = st.columns(2)
year = col1.number_input("Year", min_value=2023, max_value=2100, value=dt.date.today().year)
month = col2.selectbox("Month", list(range(1,13)), format_func=lambda m: dt.date(2000,m,1).strftime("%B"))
month_date = month_start(year, month)

title = st.text_input("Quiz title", value=f"{dt.date(2000,month,1).strftime('%B')} {year} — {cat}")
if st.button("Create/Update Quiz"):
    quiz = upsert_quiz(month_date, cat_name_to_id[cat], title)
    st.success(f"Quiz ready (id={quiz['id']}). Add questions below.")

# Fetch or create current quiz ref
quiz_row = supabase.table("quizzes").select("*").eq("month", str(month_date)).eq("category", cat_name_to_id[cat]).maybe_single().execute().data

if not quiz_row:
    st.info("Create the quiz above first.")
    st.stop()

st.subheader("Add questions from a news item")
with st.expander("Paste a news item"):
    ni_title = st.text_input("News Title")
    ni_summary = st.text_area("Summary (1–3 sentences)")
    ni_url = st.text_input("Source URL")
    if st.button("AI: Draft MCQ"):
        news_item = {"title": ni_title, "summary": ni_summary, "url": ni_url}
        draft = call_llm(news_item, f"{dt.date(2000,month,1).strftime('%B')} {year}")
        if draft:
            st.session_state["draft"] = draft

# Editor for the AI draft
draft = st.session_state.get("draft")
if draft:
    st.subheader("Review & edit draft")
    prompt = st.text_area("Question", value=draft["prompt"])
    options = []
    for i,opt in enumerate(draft["options"]):
        options.append(st.text_input(f"Option {i+1}", value=opt, key=f"opt{i}"))
    correct_idx = st.selectbox("Correct option", [0,1,2,3], index=draft["correct_index"], format_func=lambda i: f"{i+1}")
    hint = st.text_input("Hint", value=draft.get("hint",""))
    expl = st.text_area("Explanation", value=draft.get("explanation",""))
    src = st.text_input("Source URL", value=draft.get("source_url",""))

    if st.button("Save question"):
        clean = {
            "prompt": prompt.strip(),
            "options": [o.strip() for o in options],
            "correct_index": correct_idx,
            "hint": hint.strip(),
            "explanation": expl.strip(),
            "source_url": src.strip()
        }
        # store hint inside explanation field OR (better) add a column if you want hints persisted separately
        # Option A: store hint in questions.explanation as a footer
        clean_for_db = {
            "prompt": clean["prompt"],
            "explanation": clean["explanation"] + (f"\n\n(Hint: {clean['hint']})" if clean['hint'] else ""),
            "options": clean["options"],
            "correct_index": clean["correct_index"],
            "source_url": clean["source_url"]
        }
        insert_q_and_choices(quiz_row["id"], clean_for_db)
        st.success("Question saved.")
        del st.session_state["draft"]

st.subheader("Publish")
pub = st.toggle("Published", value=quiz_row.get("published", False))
if st.button("Update publish status"):
    supabase.table("quizzes").update({"published": pub}).eq("id", quiz_row["id"]).execute()
    st.success(f"Publish status updated to {pub}.")
