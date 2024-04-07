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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

def get_response_given_dict(raw_llm_output: dict) -> str:
    """Extract the response from the Groq API output using the full chain"""
    return raw_llm_output["response"]

def main():

    qdrant_client = qc.QdrantClient("https://22947c02-0f88-4954-9d59-e8fe9117b2d1.us-east4-0.gcp.cloud.qdrant.io", api_key=st.secrets['QDRANT_API_KEY'])
    collection_name = 'Taylor_Song_DataBase_full_lyrics'
    grade_collection_name = 'Grades_collection'

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device}
    )
    
    st.set_page_config(page_title="Taylor's Tune", page_icon="🎵")
    st.title("Taylor's Tune")

    st.image("C:/Users/dng09/Desktop/Project/Taylor-s-Tune/media/ts-wallpaper.webp")
    st.subheader("Find the best Taylor Swift song based on your mood", divider="rainbow", anchor=False)

    # Initialize chat history and selected model
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Define model details
    models = {
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
        "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
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
        # Layout for model selection and max_tokens slider
        col1, col2 = st.columns(2)
        with col1: 
            conv_mem_length = st.slider(
                "Memory Length:",
                min_value=1,
                max_value=10,
                value=5,
                help="Adjust the conversational memory length for the chatbot. This will affect the context of the conversation."
            )
        with col2:
            # Adjust max_tokens slider dynamically based on the selected model
            max_tokens = st.slider(
                "Max Tokens:",
                min_value=512,  # Minimum value to allow some flexibility
                max_value=max_tokens_range,
                # Default value or max allowed if less
                value=min(32768, max_tokens_range),
                step=512,
                help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
            )
        
        st.button("Clear Chat History", on_click=lambda: [st.session_state.messages.clear(), st.toast('Chat history cleared 🧹')])

    # Initializing conversation memory with the selected length
    memory=ConversationBufferWindowMemory(k=conv_mem_length)  

    # session state variable for storing chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]  # Initializing chat history if not present
    else:
        for message in st.session_state.chat_history:
            memory.chat_memory.add_user_message(message['input'])  # Saving previous chat context to memory
            memory.chat_memory.add_ai_message(message['response'])
    
    # Set up the Groq client 
    client = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name=model_option,
        max_tokens=max_tokens
    )

    # Set up conversation chain with memory buffer
    conversation = ConversationChain(
        llm=client,
        memory=memory,
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
            # Define a prompt template with specific task instructions
            prompt_template_score = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content = ("""
                                   You are an AI assistant that has to detect the score for each criteria from the user's input. The scores are explained below:
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

                                   Criteria 6: Future prospects
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
                                   
                                   This is your only goal. Don't try to do anything else.
                                   If the user input is not clear, you have to ask the user to provide more details. 
                                   Like explaining what he/she is feeling or provide a specific episode that is related to the user mood.
                                   If the user ask you something else, or ask for a clarification, you have just to explain what is your goal.

                                   You should return:
                                   - Only the score of 8 criterias. Give the score as a list of 8 numbers corresponding to each score, seperated by a comma. The list should begin with a square bracket and also end with a square bracket. No explanation before or after needed. Remember, the scores need to be a number between -3 and 3, no other symbols are allowed.
                                   - Nothing more than that. Just the score in the format above only. This conversation should not be influenced other questions or prompts.

                        """)
                    ),
                    HumanMessagePromptTemplate.from_template("{text}")
                ]
            )


            # Insert the user input into the prompt template
            human_input = prompt
            prompt_score = prompt_template_score.format_messages(text=human_input)
            # Send the prompt to the conversation chain
            
            message_score = conversation.invoke(prompt_score)

            ai_scores = get_response_given_dict(message_score)
            print(ai_scores)
            query_grades = [int(x.strip()) for x in ai_scores.split('[')[1].split(']')[0].split(',')]
            query_grades.insert(0, sum(query_grades[:4]))
            query_grades.insert(1, sum(query_grades[5:]))

            score_res = qdrant_client.search(
                collection_name = grade_collection_name,
                query_vector = query_grades,
                limit=5
            )

            query_text = embed_model.embed_documents([human_input])[0]
            song_res = qdrant_client.search(
                collection_name = collection_name,
                query_vector = query_text,
                limit=5
            )

            song_from_scores_db = "\n".join([ song.payload['metadata']['song_name'] for song in score_res])
            print(f"Song from score db: \n{song_from_scores_db}\n")

            song_from_song_db = "\n".join([ song.payload['metadata']['song_name'] for song in song_res])
            print(f"Song from lyrics db: \n{song_from_song_db}\n")
            
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content = (f"""
                                   You are an AI assistant that has two goals: detecting the user mood and suggest a Taylor Swift song compatible with the user mood.
                                   First of all you have to highlight a maximum of 5 keywords from the user input.
                                   Then you have to tell to the user which is the most relevant feeling the user is having.
                                   Finally, based on the user mood you have to suggest a Taylor Swift song that is compatible with the user mood. Use the following context to help in your suggestion {song_from_scores_db} and {song_from_song_db}. These two contexts are from two different database. The former is based on the emotional scores measured from the user text, while the second one is based on semantic analysis of the input. In each of the context, the first song is the best fit, the last song is the least fit. Select and present some of the most suitable songs from both of the databases and make this fact clear to the user

                                   Based on the user prompt try to assume to be the user and try to answer the following 6 questions giving a score from 1 to 7 for each one.

                                   For these first four questions, if you are in a relationship, answer them with respect to your current relationship. If you are not currently in a relationship, answer them by considering either your most recent past relationship, or a potential relationship on the horizon, whichever you prefer.
    
                                   Question 1
                                    Which of these best describes your relationship?
                                    1 - Our relationship ended because of cataclysmic past offenses. OR Our relationship has some serious problems.
                                    2 - My feelings were a bit hurt when our relationship ended. OR Our relationship is going ok but has some problems.
                                    3 - Our relationship ended, but not in a horribly bad way. It just ended. OR I feel pretty mediocre about the quality of our relationship.
                                    4 - I wish I was in a relationship, but I don't think it will happen right now. OR I'm happy without a relationship right now.
                                    5 - My relationship is pretty casual at the moment, not official or anything. OR I look back fondly on my past relationship, without feeling hurt or angry.
                                    6 - My relationship is going well and we're thinking about long-term commitment.
                                    7 - I'm getting married and/or comitting to this relationship for the rest of my life.
                                
                                   Question 2
                                   What does the future of your relationship look like?
                                    1 - We're never speaking again.
                                    2 - We're probably going to see each other again at some point, but we won't be in touch much at all.
                                    3 - We might talk a bit less than we did in the past.
                                    4 - I'm not sure what our future is.
                                    5 - We've got some casual future plans but nothing serious lined up. OR We might hang out but I'm not sure.
                                    6 - We're going to be spending a fair amount of time together in the future.
                                    7 - We're going to be spending a large amount of time together. Like maybe getting married.
                                
                                   Question 3
                                   	What are the other person's feelings about you?
                                    1 - They've told me they hate me.
                                    2 - I think they don't like me that much. OR They've insulted me some.
                                    3 - They're nice to me but they see me as just a friend.
                                    4 - I'm not sure and/or they haven't made it clear to me.
                                    5 - They maybe have some non-platonic feelings for me but I'm not sure how strong they are.
                                    6 - They've told me that they have some feelings for me.
                                    7 - They have openly declared their love for me to the world.
                                   
                                   Question 4
                                   	Which of these best describes how you spend your time together?
                                    1 - There are significant barriers that prevent us from being together.
                                    2 - There aren't any insurmountable barriers between us, but we never do anything together.
                                    3 - We do some things together but spend most of our time doing things alone.
                                    4 - We do about the same amount of stuff together as we do alone.
                                    5 - We do some things alone but spend most of our time doing things together.
                                    6 - We do pretty much everything together.
                                    7 - We do everything together, and even when we aren't together I only think about us being together.

                                    For these next two questions, think about how you feel about your life overall.
                                    Question 5
                                    Which of these best describes how you feel about yourself?
                                    1 - I have a lot of problems and they're all my fault.
                                    2 - I have a lot of problems, but I don't think they're all my fault.
                                    3 - I don't have a ton of significant problems, but sometimes I think I could do better.
                                    4 - I'm not really sure how I feel.
                                    5 - I feel pretty good about myself, and am just a little insecure on occasion.
                                    6 - I have a few concerns but feel very good overall.
                                    7 - I'm awesome, my life is awesome, this is the bomb.
                                   
                                   Question 6
                                   Which of these describes your emotional state?
                                    1 - You're really angry about something and/or really depressed about something.
                                    2 - You don't like how your life is going and you just want to make a deal to get your old life back.
                                    3 - You know something's wrong with your life but you want to ignore it.
                                    4 - You've accepted the bad things that have happened to you and are ready to move on from them.
                                    5 - You're feeling pretty neutral and you're waiting for life to make you happy.
                                    6 - You're actively working to make yourself happy.
                                    7 - You're actively working to make yourself happy and trying to make sure that everyone else is happy too.

                                   This is your only goal. Don't try to do anything else.
                                   If the user input is not clear, you have to ask the user to provide more details. 
                                   Like explaining what he/she is feeling or provide a specific episode that is related to the user mood.
                                   If the user ask you something else, or ask for a clarification, you have just to explain what is your goal.

                                   You should return:
                                   - A message that contains the most relevant feeling the user is having.
                                   - The most suitable songs from the emotional score database and the lyrics semantic database, make it clear to the user what each database represents. Do not include the score of the songs.
                                   - The number of the answer for each question. The answer should be formatted in a way that each question has its own line with the question, the score, and the reasoning behind that score.
                                 Do not inlcude the scores predicted in the previous prompt to the answer. Those criteria are only to extract the context from the database and they are not needed for the answer here
                        """)
                    ),
                    HumanMessagePromptTemplate.from_template("{text}")
                ]
            )

            prompt = prompt_template.format_messages(text=prompt)
            message = conversation.invoke(prompt)

            # Store the message in the chat history
            st.session_state.chat_history.append(message)  

            # Use the generator function with st.write_stream
            with st.chat_message("assistant", avatar="🤖"):
                ai_responses = get_response_given_dict(message)
                st.write(ai_responses)
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages
        if isinstance(ai_responses, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_responses})
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in ai_responses)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response})

if __name__ == "__main__":
    main()