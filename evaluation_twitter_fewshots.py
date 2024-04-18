import numpy as np
import pandas as pd
from typing import Generator
from langchain.chains import ConversationChain 
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  
from langchain_groq import ChatGroq  
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate  
import re

import time
from utils_metrics import cosine_similarity, euclidean_distance

import pickle

n_samples = 3000
emoint_df = pd.read_csv(f'data/cleaned_data/emoint_test_{n_samples}_sample.csv', index_col=0)
sampled_df = emoint_df.sample(3, replace=False)

CONTENT = f"""
        You are an AI assistant that has one goal: detecting the user emotions from the tweet input.
        You have to detect the emotions and score the intensity of 4 emotions: anger, fear, joy, sadness from 0 to 1.
        
        For example:

        If the input is: {sampled_df.index.values[0]}, then the output should be: {'['+', '.join(sampled_df.iloc[0].astype(str).values)+']'}.
        If the input is: {sampled_df.index.values[1]}, then the output should be: {'['+', '.join(sampled_df.iloc[1].astype(str).values)+']'}.
        If the input is: {sampled_df.index.values[2]}, then the output should be: {'['+', '.join(sampled_df.iloc[2].astype(str).values)+']'}.
        

        This is your only goal. Don't try to do anything else.

        You should ONLY return:
        - A vector of 4 numbers from 0 to 1 in a vector opened and closed by square brackets, corresponding to the intensity of each emotion found in the user input.
        - Before the vector, you should return the word "emotions:" to indicate that the following vector is the emotion intensity vector.
        """

print(CONTENT)

def main():
    # Consider the statistical model as ground truth 
    # Compare before and after the RAG model

    
    
    # randomly sample 300 rows
   

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


    
    print("="*50)
    print("Model: ", model["name"])

    max_tokens = model["tokens"]

   

    for i, prompt in enumerate(emoint_df.index.values):
        
        label = list(emoint_df.iloc[i].values)
        
        
        

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

    # print(predicted_list)
    # print(label_list)

    with open(f'fewshots_emoint_dev_{n_samples}_predicted_list.pkl', 'wb') as f:
        pickle.dump(predicted_list, f)

    with open(f'fewshots_emoint_dev_{n_samples}_label_list.pkl', 'wb') as f:
        pickle.dump(label_list, f)

if __name__ == "__main__":
    main()

