import numpy as np
import pandas as pd
from typing import Generator
from langchain.chains import ConversationChain 
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  
from langchain_groq import ChatGroq  
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate  
import re
import pickle
import time
from utils_metrics import cosine_similarity, euclidean_distance

CONTENT = """
        You are an AI assistant that has one goal: detecting the user emotions from the text input.
        You have to detect the emotion out of 6 emotions anger,fear,joy,love,sadness,surprise.

        This is your only goal. Don't try to do anything else.

        You should ONLY return:
            - The vector with 6 numbers for the intensity between 0 and 1 of the emotion detected in the text input for the emotion:  anger, fear, joy, love, sadness, surprise.
            - The vector should be opened and closed by square brackets.
"""

print(CONTENT)

def main():
    # Consider the statistical model as ground truth 
    # Compare before and after the RAG model

    emotion_df = pd.read_csv('data/cleaned_data/kaggle_emotion_1000_sample.csv', index_col=0)
    # emotion_df = emotion_df.sample(300, replace=False)
    predicted_list = []
    label_list = []

    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_response_given_dict(raw_llm_output: dict) -> str:
        """Extract the response from the Groq API output using the full chain"""
        return raw_llm_output["response"]

    model = {'id':"mixtral-8x7b-32768", "name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}


    
    # print("="*50)
    # print("Model: ", model["name"])

    max_tokens = model["tokens"]
    try:
        for i, prompt in enumerate(emotion_df.index.values):
        
        # print(prompt)
            label = list(emotion_df.iloc[i].values)
            client = ChatGroq(
                # api_key=st.secrets["GROQ_API_KEY"],
                api_key = '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', # local testing
                model_name=model["id"],
                max_tokens=max_tokens
            )

            memory=ConversationBufferWindowMemory(k=5)  

            conversation = ConversationChain(
                llm=client,
                memory=memory,
            )

            prompt_template = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(
                            content = (CONTENT)        
                        ),

                        HumanMessagePromptTemplate.from_template("{text}")

                    ]
                )
            
            prompt = prompt_template.format_messages(text=prompt)
            message = conversation.invoke(prompt)

            ai_responses = get_response_given_dict(message)
            # print(ai_responses)
            try:
                match = re.search(r'\[.*?\]', ai_responses)
                ai_emotion_vector = eval(match.group(0))

                predicted_list.append(ai_emotion_vector)
                label_list.append(label)
            except Exception as e:
                print(e)
                print('Prompt:', prompt)
                print('AI Response:', ai_responses)
                print('\n')
                continue

            time.sleep(2) # To meet the API rate limit
        # # print(predicted_list)
        # # print(label_list)

            
        
    except Exception as e:
        print(e)
    
    finally:
        with open('kaggle_emotion_1000_predicted_list.pkl', 'wb') as f:
            pickle.dump(predicted_list, f)

        with open('kaggle_emotion_1000_label_list.pkl', 'wb') as f:
            pickle.dump(label_list, f)

if __name__ == "__main__":
    main()

