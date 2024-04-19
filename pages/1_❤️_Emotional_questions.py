import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ts_questionaire import *

st.set_page_config(page_title="Emotional Questions", page_icon="❓")

st.markdown("# ❓ Emotional Questions")
st.markdown("This work was inspired by the paper *[I Knew You Were Trouble: Emotional Trends in the Repertoire of Taylor Swift](https://arxiv.org/abs/2103.16737)*")
st.markdown("## Instructions")
st.markdown("This page simulates the usage of the [`taylorswift` library](https://pypi.org/project/taylorswift/), which is a Python package that provides song suggestions based on emotional criteria. (No fancy LLMs here, just some good old statistics tools!)")
st.markdown("*Answer to each question in order to obtain songs suggestions*")

st.markdown("## Question 1")
# st.markdown("### Which of these best describes your relationship?\n")
q1 = """1 - Our relationship ended because of cataclysmic past offenses. OR Our relationship has some serious problems.
2 - My feelings were a bit hurt when our relationship ended. OR Our relationship is going ok but has some problems.
3 - Our relationship ended, but not in a horribly bad way. It just ended. OR I feel pretty mediocre about the quality of our relationship.
4 - I wish I was in a relationship, but I don't think it will happen right now. OR I'm happy without a relationship right now.
5 - My relationship is pretty casual at the moment, not official or anything. OR I look back fondly on my past relationship, without feeling hurt or angry.
6 - My relationship is going well and we're thinking about long-term commitment.
7 - I'm getting married and/or comitting to this relationship for the rest of my life."""


q1_q = st.radio(
    "Which of these best describes your relationship?",
    q1.split("\n"),
    index=None,
)
st.write("For Q1 you selected:", q1_q)

st.markdown("## Question 2")
# st.markdown("### What does the future of your relationship look like?\n")
q2 = ("""1 - We're never speaking again.
2 - We're probably going to see each other again at some point, but we won't be in touch much at all.
3 - We might talk a bit less than we did in the past.
4 - I'm not sure what our future is.
5 - We've got some casual future plans but nothing serious lined up. OR We might hang out but I'm not sure.
6 - We're going to be spending a fair amount of time together in the future.
7 - We're going to be spending a large amount of time together. Like maybe getting married.""")
q2_q = st.radio(
    "What does the future of your relationship look like?",
    q2.split("\n"),
    index=None,
)
st.write("For Q2 you selected:", q2_q)

st.markdown("## Question 3")
# st.write("""### What are the other person's feelings about you?\n""")
q3 = ("""1 - They've told me they hate me.
2 -  think they don't like me that much. OR They've insulted me some.
3 - They're nice to me but they see me as just a friend.
4 - I'm not sure and/or they haven't made it clear to me.
5 - They maybe have some non-platonic feelings for me but I'm not sure how strong they are.
6 - They've told me that they have some feelings for me.
7 - They have openly declared their love for me to the world.""")

q3_q = st.radio(
    "What are the other person's feelings about you?",
    q3.split("\n"),
    index=None,
)
st.write("For Q3 you selected:", q3_q)

st.markdown("## Question 4")
# st.markdown("### Which of these best describes how you spend your time together?\n")
q4 = ("""1 - There are significant barriers that prevent us from being together.
2 - There aren't any insurmountable barriers between us, but we never do anything together.
3 - We do some things together but spend most of our time doing things alone.
4 - We do about the same amount of stuff together as we do alone.
5 - We do some things alone but spend most of our time doing things together.
6 - We do pretty much everything together.
7 - We do everything together, and even when we aren't together I only think about us being together.""")

q4_q = st.radio(
    "Which of these best describes how you spend your time together?",
    q4.split("\n"),
    index=None,
)
st.write("For Q4 you selected:", q4_q)

st.markdown("## Question 5")
# st.markdown("### Which of these best describes how you feel about yourself?\n")
q5 = ("""1 - I have a lot of problems and they're all my fault.
2 - I have a lot of problems, but I don't think they're all my fault.
3 - I don't have a ton of significant problems, but sometimes I think I could do better.
4 - I'm not really sure how I feel.
5 - I feel pretty good about myself, and am just a little insecure on occasion.
6 - I have a few concerns but feel very good overall.
7 - I'm awesome, my life is awesome, this is the bomb.""")

q5_q = st.radio(
    "Which of these best describes how you feel about yourself?",
    q5.split("\n"),
    index=None,
)
st.write("For Q5 you selected:", q5_q)

st.markdown("## Question 6")
q6 = ("""1 You're really angry about something and/or really depressed about something.
2 - You don't like how your life is going and you just want to make a deal to get your old life back.
3 - You know something's wrong with your life but you want to ignore it.
4 - You've accepted the bad things that have happened to you and are ready to move on from themn
5 - You're feeling pretty neutral and you're waiting for life to make you happy
6 - You're actively working to make yourself happy
7 - You're actively working to make yourself happy and trying to make sure that everyone else is happy too.""")

q6_q = st.radio(
    "Which of these best describes your emotional state?",
    q6.split("\n"),
    index=None,
)
st.write("For Q6 you selected:", q6_q)

if q1_q == None or q2_q == None or q3_q == None or q4_q == None or q5_q == None or q6_q == None:
    st.warning("Please answer all questions")
else:
    answers = [q1_q[0], q2_q[0], q3_q[0], q4_q[0], q5_q[0], q6_q[0]]
    answers = list(map(int, answers))
    st.write(f"Your answers are: {answers}")
    songs = get_songs(answers)
    urls = [find_match(s) for s in songs]
    for s, u in zip(songs, urls):
        st.write(f"- {s} ({u})")