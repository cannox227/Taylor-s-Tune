import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Generator
from langchain.chains import ConversationChain 
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  
from langchain_groq import ChatGroq  
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
import qdrant_client as qc
from qdrant_client.http.models import *
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import logging
from ts_questionaire import *
from data.prompts import evaluation_prompts

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
evaluation_mode = False

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

def get_response_given_dict(raw_llm_output: dict) -> str:
    """Extract the response from the Groq API output using the full chain"""
    return raw_llm_output["response"]

def main():

    qdrant_client = qc.QdrantClient(st.secrets['QDRANT_CLIENT_URL'], api_key=st.secrets['QDRANT_API_KEY'])
    collection_name = 'Taylor_Song_DataBase_full_lyrics'
    grade_collection_name = 'Grades_collection'

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device}
    )
    
    criteria = ['feelings of self', 'glass half full', 'stages of depression', 'tempo', 'seriousness', 'future prospects', 'feeling of male', 'togetherness']

    st.set_page_config(page_title="Taylor's Tune", page_icon="🎵")
    st.title("Taylor's Tune")

    st.image("media/ts-wallpaper.webp")
    st.subheader("Find the best Taylor Swift song based on your mood", divider="rainbow", anchor=False)

    st.markdown("*Try to explain how do you feel and your relationship status using the chat and the agent will help you finding the most suited Taylor Swift songs for you!*")
    # Initialize chat history and selected model
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Define model details
    models = {
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
        "llama3-70b-8192": {"name": "llama3-70b-8192", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "llama3-8b-8192", "tokens": 8192, "developer": "Meta"}, 
        "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"}
    }

    # Sidebar customization
    st.sidebar.title("Customization")
    with st.sidebar:
        # Model selection
        model_option = st.selectbox(
                "Choose a model:",
                options=list(models.keys()),
                format_func=lambda x: models[x]["name"],
                index=0  # Default to the first model in the list
            )

        # Detect model change and clear chat history if model has changed
        if st.session_state.selected_model != model_option:
            st.session_state.messages = []
            st.session_state.selected_model = model_option

        max_tokens_range = models[model_option]["tokens"]

        qdrant_query_limit = st.slider(min_value=1, max_value=5, value=3, label="Number of song suggestions", help="Select the maximum number of songs that will be suggest based on your mood")
        # Layout for model selection and max_tokens slider
        col1, col2 = st.columns(2)
        with col1: 
            conv_mem_length = st.slider(
                "Memory Length:",
                min_value=1,
                max_value=5,
                value=1,
                help="Adjust the conversational memory length for the chatbot. This will affect the context of the conversation."
            )
        with col2:
            # Adjust max_tokens slider dynamically based on the selected model
            max_tokens = st.slider(
                "Max Tokens:",
                min_value=512,  # Minimum value to allow some flexibility
                max_value=max_tokens_range,
                # Default value or max allowed if less
                value=2048,
                step=512,
                help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
            )
        
        st.button("Clear Chat History", on_click=lambda: [st.session_state.messages.clear(), st.toast('Chat history cleared 🧹')])

    # Initializing conversation memory with the selected length
    # memory=ConversationBufferWindowMemory(k=conv_mem_length)  

    # session state variable for storing chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]  # Initializing chat history if not present
    else:
        # TODO: enable again history
        # for message in st.session_state.chat_history:
        #     memory.chat_memory.add_user_message(message['input'])  # Saving previous chat context to memory
        #     memory.chat_memory.add_ai_message(message['response'])
        pass
    
    # Set up the Groq client 
    client = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name=model_option,
        max_tokens=max_tokens
    )

    # Set up conversation chain with memory buffer
    conversation = ConversationChain(
        llm=client,
        #memory=memory,
    )

    

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Get user input and send to Groq API
    if prompt := st.chat_input("How do you feel today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)

        # Fetch response from Groq API
        try:
            st.toast("Processing your input... 🤖")
            # Define a prompt template with specific task instructions
            
            # PRE-PROMPT 1
            
            prompt_template_score = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content = ("""
You are an AI assistant that has to complete 2 tasks.
---
Task 1: 
detect the score for each criteria from the user's input. 
The scores are explained below:
Criteria 1: Feelings of self
-3 - Feels fully responsible for problems
-2 - Feels partial responsibility for problems 
-1 - Hints at self-deprecation 
0  - No feelings mentioned/ambiguous feelings 
1  - Overall positive with serious insecurities 
2  - Overall positive with some reservations
3  - Secure and trusting in life circumstances 

Criteria 2: Glass half full
-3 - All imagery is depressing 
-2 - Nearly all depressing imagery  
-1 - Majority depressing imagery
0  - Equal amounts of happy and sad imagery  
1  - Majority positive imagery
2  - Nearly all positive imagery
3  - All imagery is positive 

Criteria 3: Stages of depression
-3 - Anger / Depression
-2 - Bargaining
-1 - Denial
0  - Acceptance. If you don't know what to give, just give this score
1  - Passively wanting to be happy 
2  - Actively working for her happiness 
3  - Actively working for her own and others' happiness

Criteria 4: Tempo
0 - No tempo, this is not a song

Criteria 5: Seriousness
-3 - Cataclysmic past offenses 
-2 - Some past hurt feelings
-1 - Unspecified relationship endings
0  - Not discussed/Pining
1  - Puppy love/One night stand 
2  - Some real world things to discuss
3  - Discussion of marriage/equally serious topics

-3 - Permanent end to communication 
-2 - Significant decrease in contact 
-1 - Possible decrease in contact 
0  - No discussion of future/Ambiguous 
1  - Casual or potential future plans  
2  - Some set future plans
3  - Marriage/Bound for life 

Criteria 7: Feelings of males
-3 - He tells all his friends he hates her
-2 - He makes a face when her name is mentioned but doesn't publicly hate on her 
-1 - He doesn't want to date but likes her as a friend
0  - No information/Ambiguous. If you're not sure, also give this score
1  - He expressed casual interest in a relationship
2  - They are dating but not that seriously (she hasn't met his parents)
3  - Public declaration of love/commitment

Criteria 8: Togetherness
-3 - Barriers to joint actions 
-2 - No joint actions 
-1 - More things apart than together 
0  - Equal amounts of time together and apar
1  - More things together than apart 
2  - They do everything together
3  - No identity as an individual 

If you think the criteria are not applicable in the situation. Give the score 0.
---- 
This is your only goal. Don't try to do anything else.
If the user input is not clear, you have to ask the user to provide more details. 
Like explaining what he/she is feeling or provide a specific episode that is related to the user mood.
If the user ask you something else, or ask for a clarification, you have just to explain what is your goal.
If the user ask you for something missing from the previous prompt you have to ask the user to provide the missing information. Do not make up missing information!
You should return:
- As first output the score of 8 criterias. Give the score as a list (called criteria list) of 8 numbers corresponding to each score, seperated by a comma. The list should begin with a square bracket and also end with a square bracket. No explanation before or after needed. Remember, the scores need to be a number between -3 and 3, no other symbols are allowed. The criteria list must have length 8, a different lenght is not allowed.
Do not forget to enclose the criteria list in square brackets. Do not send it as a list of numbers separated by commas without square brackets.
- For each element of the list just one numerical value from -3 to 3 should be present, no other words or interadiate values are allowed.
- Here is an example of the format you should use:
The criteria list is: [3, 3, 0, 0, 0, 3, 0, 0]
- Always report the criteria lists. Do not forget to report it in the proper format.
- Nothing more than that. Just the score in the format above only. This conversation should not be influenced other questions or prompts.
                        """)
                    ),
                    HumanMessagePromptTemplate.from_template("{text}")
                ]
            )

            if evaluation_mode:
                prompt_template = evaluation_prompts.evaluation_prompt
            
            # Insert the user input into the prompt template
            human_input = prompt
            prompt_score = prompt_template_score.format_messages(text=human_input)
            # Send the prompt to the conversation chain
             
            message_score = conversation.invoke(prompt_score)

            ai_tasks_reply = get_response_given_dict(message_score)

            logger.info(f"\nAI tasks reply: {ai_tasks_reply}")

            # TODO: try to understand if here the LLM understood or not the prompt
            # TODO: handle tasks like this ans skip the second stage
            # AI tasks reply: I'm sorry to hear that you're feeling this way, but I'm unable to provide scores for the criteria or answers to the questions as you've requested. However, I can provide some song suggestions based on your current feelings.

            # 1. "the outside" - This song might resonate with your feelings of being on the outside looking in. Spotify link: https://open.spotify.com/track/2QA3IixpRcKyOdG7XDzRgv
            # 2. "a place in this world" - This song fits your sentiment of trying to find your place. Spotify link: https://open.spotify.com/track/73OX8GdpOeGzKC6OvGSbsv
            # 3. "how you get the girl" - This song might resonate with your hope of meeting someone new. Spotify link: https://open.spotify.com/track/733OhaXQIHY7BKtY3vnSkn
            # 4. "right where you left me" - This song matches your current emotional state of being left behind. Spotify link: https://open.spotify.com/track/3zwMVvkBe2qIKDObWgXw4N
            # 5. "invisible" - This song talks about feeling unnoticed and unappreciated, which might be a part of your current emotional experience. Spotify link: https://open.spotify.com/track/5OOd01o2YS1QFwdpVLds3r

            # Remember, it's okay to feel sad and it's okay to hope for something new. Take your time to heal and when you're ready, the right person will come into your life.


            logger.info(f"Criteria scores from first LLM output: ")
            evaluation_error = False
            try:
                criterion_grades = [int(x.strip()) for x in ai_tasks_reply.split('[')[1].split(']')[0].split(',')]
                assert len(criterion_grades) == 8, "Criteria list must have length 8"
            except Exception as e:
                st.toast('An error occurred while processing the user input. Please try again.')
                logger.error(f"Error parsing criteria scores: {e}")
                criterion_grades = [0, 0, 0, 0, 0, 0, 0, 0]
                evaluation_error = True
            
            if evaluation_mode:
                try:
                    questions_grades = [int(x.strip()) for x in ai_tasks_reply.split('{')[1].split('}')[0].split(',')] 
                    assert len(questions_grades) == 6, "Questions list must have length 8"
                except Exception as e:
                    st.toast('An error occurred while processing the user input. Please try again.')
                    logger.error(f"Error parsing questions scores: {e}")
                    questions_grades = [0, 0, 0, 0, 0, 0]
                    evaluation_error = True
                
                if criterion_grades == [0, 0, 0, 0, 0, 0, 0, 0] or questions_grades == [0, 0, 0, 0, 0, 0]:
                    evaluation_error = True

            if not evaluation_error:
                logger.info(f"Criteria grades: {criterion_grades}")
                if evaluation_mode: logger.info(f"Questions grades: {questions_grades}")

                if logger.level == logging.DEBUG:
                    for idx, score in enumerate(criterion_grades):
                        logger.debug(f"{criteria[idx]}: {score}") 
                    if evaluation_mode:
                        for idx, score in enumerate(questions_grades):
                            logger.debug(f"Question {idx+1}: {score}")

                criterion_grades.insert(0, sum(criterion_grades[:4]))
                criterion_grades.insert(1, sum(criterion_grades[5:]))

                score_res = qdrant_client.search(
                    collection_name = grade_collection_name,
                    query_vector = criterion_grades,
                    limit=qdrant_query_limit
                )

                query_text = embed_model.embed_documents([human_input])[0]
                song_res = qdrant_client.search(
                    collection_name = collection_name,
                    query_vector = query_text,
                    limit=qdrant_query_limit
                )

                song_from_scores_db = ""
                song_from_song_db = ""

                for song_score, song_lyrics in zip(score_res, song_res):
                    song_from_scores_db += f"- {song_score.payload['metadata']['song_name']}, Spotify link: {song_score.payload['metadata']['url']}, lyrics: {song_score.payload['metadata']['description']}\n"
                    song_from_song_db += f"- {song_lyrics.payload['metadata']['song_name']}, Spotify link: {song_lyrics.payload['metadata']['url']}, lyrics: {song_lyrics.payload['metadata']['description']}\n"

                logger.info(f"\nSong from score db: \n{song_from_scores_db}\n")

                logger.info(f"\nSong from lyrics db: \n{song_from_song_db}\n")

                if evaluation_mode:
                    lib_predicted_songs = "\n".join(get_songs(questions_grades))
                    logger.info(f"Predicted songs: {lib_predicted_songs}")

                # Prompt template for the AI response
                # - Songs retrieved by a statistical model: {lib_predicted_songs}\n. 
                
                # PRE-PROMPT 2
                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(
                            content = (f"""
You are an AI assistant that has two goals: detecting the user mood and suggest a Taylor Swift song compatible with the user mood.
First of all you have to highlight a maximum of 5 keywords from the user input.
Then you have to tell to the user which is the most relevant feeling the user is having.
Finally, based on the user mood you have to suggest a Taylor Swift song that is compatible with the user mood. 
Use the following context to help in your suggestion:
- Songs retrieved by user emotion analysis: {song_from_scores_db}\n
- Songs retrieved according to lyrics matching: {song_from_song_db}\n
The first item of each pair is the song name, the second item of the pair is the Spotify link of the song and the third is the song description. These contexts are from two different database. The former is based on the emotional scores measured from the user text, while the second one is based on semantic analysis of the input. In each of the context, the first song is the best fit, the last song is the least fit. 
Present all the songs for each database and the corresponding Spotify link (the item after the colon) from both of the databases and make this fact clear to the user. 
Remember to include and show all the resulted songs and avoid mixing the result of one db with the other! {qdrant_query_limit*2} songs must be showed in total.

You should return a reponse that includes the following information:
- Your understanding of the user mood in natural language.
- Say to the user based on its input a maximum of 5 keywords that you detected.
- Say what are the main emotions or the general feelings perceived by the user message (a maximum of three and a minimum of one).
- All the {qdrant_query_limit} suggested songs from the emotion analysis.
- All the {qdrant_query_limit}suggested song based on the lyrics matching.
- Conclude the message with your personal suggestion on how to deal with the sentiments the user is feeling.

For each songs result, you should present the song name, the Spotify link and the given description of the meaning of that song.
Do not forget the Spotiky link!  If the Spotify link is not available or equal to "-" just write "Spotify link not available". Do not make up the Spotify link. 
Always answer in natural language and avoid to just report the data you have in the database (avoid to just copy and paste the data from the database with their keywords).

You must not return:
- Any previous criteria scores or the scores of the questions. Those criteria and questions are only to extract the context from the database and they are not needed for the answer here.
""")
                        ),
                        HumanMessagePromptTemplate.from_template("{text}")
                    ]
                )
                
                prompt = prompt_template.format_messages(text=prompt)
                conversation.memory.clear()
                logger.info(f"PROMPT USING SECOND PREPRPOMPT {prompt}")
                try:
                    message = conversation.invoke(prompt)
                except Exception as e:
                   with st.chat_message("assistant", avatar="🤖"):
                        st.write(f"High request rate please wait\n{e}") 
                logger.info(f"AI response DEBUG: {message}")
                ai_responses = get_response_given_dict(message)
            else:
                ai_responses = ai_tasks_reply
                st.error(ai_responses)
                st.error("😕 An error occurred while processing the user input.")
            # Store the message in the chat history
            if not evaluation_error: st.session_state.chat_history.append(message)  

            # Use the generator function with st.write_stream
            with st.chat_message("assistant", avatar="🤖"):
                if not evaluation_error: st.write(ai_responses)
                else: st.write("Please try to be more clear 😢🙏")
        except Exception as e:
            st.error(f'{type(e).__name__,}\n{e}', icon="🚨")
            #conversation.memory.clear()

        # Append the full response to session_state.messages
        if isinstance(ai_responses, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_responses})
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in ai_responses)
            if not evaluation_error: st.session_state.messages.append(
                {"role": "assistant", "content": combined_response}) 
    
    
if __name__ == "__main__":
    main()