import streamlit as st
from supabase import create_client

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