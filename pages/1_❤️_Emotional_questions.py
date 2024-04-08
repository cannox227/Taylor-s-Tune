import streamlit as st

st.set_page_config(page_title="Emotional Questions", page_icon="❓")

st.markdown("# ❓ Emotional Questions")
st.markdown("*This set of questions are used by the LLM to emphatize with the user in order to better understand the context of the prompt*")
st.markdown("## Question 1")
st.markdown("### Which of these best describes your relationship?\n")
st.write("""
    1 - Our relationship ended because of cataclysmic past offenses. OR Our relationship has some serious problems.\n
    2 - My feelings were a bit hurt when our relationship ended. OR Our relationship is going ok but has some problems.\n
    3 - Our relationship ended, but not in a horribly bad way. It just ended. OR I feel pretty mediocre about the quality of our relationship.\n
    4 - I wish I was in a relationship, but I don't think it will happen right now. OR I'm happy without a relationship right now.\n
    5 - My relationship is pretty casual at the moment, not official or anything. OR I look back fondly on my past relationship, without feeling hurt or angry.\n
    6 - My relationship is going well and we're thinking about long-term commitment.\n
    7 - I'm getting married and/or comitting to this relationship for the rest of my life.\n
         """)

st.markdown("## Question 2")
st.markdown("### What does the future of your relationship look like?\n")
st.write("""
    1 - We're never speaking again.\n
    2 - We're probably going to see each other again at some point, but we won't be in touch much at all.\n
    3 - We might talk a bit less than we did in the past.\n
    4 - I'm not sure what our future is.\n
    5 - We've got some casual future plans but nothing serious lined up. OR We might hang out but I'm not sure.\n
    6 - We're going to be spending a fair amount of time together in the future.\n
    7 - We're going to be spending a large amount of time together. Like maybe getting married.\n
         """)

st.markdown("## Question 3")
st.write("""### What are the other person's feelings about you?\n""")
st.write("""
    1 - They've told me they hate me.\n
    2 - I think they don't like me that much. OR They've insulted me some.\n
    3 - They're nice to me but they see me as just a friend.\n
    4 - I'm not sure and/or they haven't made it clear to me.\n
    5 - They maybe have some non-platonic feelings for me but I'm not sure how strong they are.\n
    6 - They've told me that they have some feelings for me.\n
    7 - They have openly declared their love for me to the world.\n
            """)

st.markdown("## Question 4")
st.markdown("### Which of these best describes how you spend your time together?\n")
st.write("""
         1 - There are significant barriers that prevent us from being together.\n
         2 - There aren't any insurmountable barriers between us, but we never do anything together.\n
         3 - We do some things together but spend most of our time doing things alone.\n
         4 - We do about the same amount of stuff together as we do alone.\n
         5 - We do some things alone but spend most of our time doing things together.\n
         6 - We do pretty much everything together.\n
         7 - We do everything together, and even when we aren't together I only think about us being together.\n
         """)

st.markdown("## Question 5")
st.markdown("### Which of these best describes how you feel about yourself?\n")
st.write("""
            1 - I have a lot of problems and they're all my fault.\n
            2 - I have a lot of problems, but I don't think they're all my fault.\n
            3 - I don't have a ton of significant problems, but sometimes I think I could do better.\n
            4 - I'm not really sure how I feel.\n
            5 - I feel pretty good about myself, and am just a little insecure on occasion.\n
            6 - I have a few concerns but feel very good overall.\n
            7 - I'm awesome, my life is awesome, this is the bomb.\n
            """)

st.markdown("## Question 6")
st.markdown("### Which of these describes your emotional state?\n")
st.write("""
            1 - You're really angry about something and/or really depressed about something.\n
            2 - You don't like how your life is going and you just want to make a deal to get your old life back.\n
            3 - You know something's wrong with your life but you want to ignore it.\n
            4 - You've accepted the bad things that have happened to you and are ready to move on from them.\n
            5 - You're feeling pretty neutral and you're waiting for life to make you happy.\
            6 - You're actively working to make yourself happy.\n
            7 - You're actively working to make yourself happy and trying to make sure that everyone else is happy too.\n
            """)