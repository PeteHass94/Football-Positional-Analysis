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

displaytext = """## Welcome to Football Analysis on Positional Data App \n\n"""

st.markdown(displaytext)

displaytext = (
    """Using data made open from SkillCorner.  https://github.com/SkillCorner/opendata/tree/master \n\n"""
    """This app will look to analyze football (soccer) teams and players using positional data. \n\n"""
    """The hope is to produce a reliable indicator if positional data indicates likelihood of chance creation or conceding. \n\n"""
)

st.markdown(displaytext)



