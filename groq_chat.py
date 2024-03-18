import streamlit as st
from typing import Generator
from langchain.chains import ConversationChain 
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  
from langchain_groq import ChatGroq  
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate  

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

def get_response_given_dict(raw_llm_output: dict) -> str:
    """Extract the response from the Groq API output using the full chain"""
    return raw_llm_output["response"]

def main():
    
    st.set_page_config(page_title="Taylor's Tune", page_icon="🎵")
    st.title("Taylor's Tune")

    st.image("media/ts-wallpaper.webp")
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
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content = ("""
                                   You are an AI assistant that has two goals: detecting the user mood and suggest a Taylor Swift song compatible with the user mood.
                                   First of all you have to highlight a maximum of 5 keywords from the user input.
                                   Then you have to tell to the user which is the most relevant feeling the user is having.
                                   Finally, based on the user mood you have to suggest a Taylor Swift song that is compatible with the user mood. 

                                   This is your only goal. Don't try to do anything else.
                                   If the user input is not clear, you have to ask the user to provide more details. 
                                   Like explaining what he/she is feeling or provide a specific episode that is related to the user mood.
                                   If the user ask you something else, or ask for a clarification, you have just to explain what is your goal.
                        """)
                    ),
                    HumanMessagePromptTemplate.from_template("{text}")
                ]
            )

            # Insert the user input into the prompt template
            prompt = prompt_template.format_messages(text=prompt)
            # Send the prompt to the conversation chain
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