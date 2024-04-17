import streamlit as st
import pandas as pd

st.set_page_config(page_title="Emotional Scores", page_icon="📊")

st.markdown("# 📊 Emotional Scores")

st.markdown("## Emotional criteria for each song")
st.markdown("""Each song contains 2 macro parameters:\n
- Happy_sad value: from -12 to +12\n
- Relationship value: from -12 to +12\n

These two parameters are obtained by other sub criteria such as:
*feelings of self, glass half full, stages of depression, tempo, seriousness, future prospects, feeling of male, togetherness*\n
These values have been obtained by the results of the paper *[I Knew You Were Trouble: Emotional Trends in the Repertoire of Taylor Swift](https://arxiv.org/abs/2103.16737)*
""")
criteria_df = pd.read_csv("data/lyrics_criteria.csv")
st.markdown("### Criteria values explained")
st.dataframe(criteria_df)
st.markdown("## Emotional criteria for each song")
songs_criteria_df = pd.read_csv("data/cleaned_data/rag_dataset_enhanced.csv")
st.dataframe(songs_criteria_df)