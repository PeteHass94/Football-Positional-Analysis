# import streamlit as st

# st.title("🎈 My new app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )


"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).

conda create --name streamlit_env
conda activate streamlit_env
pip install -r requirements.txt
streamlit run streamlit_app.py
"""

# Library imports
import traceback
import copy

import streamlit as st


from utils.page_components import (
        add_common_page_elements,
    )
    
sidebar_container = add_common_page_elements()

displaytext = """## Welcome to Football Analysis on Game State """

st.markdown(displaytext)

displaytext = (
    """Game state is described as a teams state in a game, whether they are winning, drawing or losing. \n\n"""
    """I will be showing how I scrape data from the internet then display in this website. """
    """I will look into a number of factors to help show important information on each team. \n\n"""
)

st.markdown(displaytext)



