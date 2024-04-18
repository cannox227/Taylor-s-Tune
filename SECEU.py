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

CONTENT = f"""
        You need to read this story and evaluate the emotion of the main character. I will give you four emotion options. 
        For each option, you need to decide how much the main character feel that emotion by rating between 0 to 10. 
        Notice that: (a)the total score of the four options should be exactly 10, (b)output JSON with key name and score.
"""

with open('seceu_item.pkl', 'rb') as f:
    seceu_item = pickle.load(f)

def main():

    res = {}
    # Consider the statistical model as ground truth 
    # Compare before and after the RAG model

    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_response_given_dict(raw_llm_output: dict) -> str:
        """Extract the response from the Groq API output using the full chain"""
        return raw_llm_output["response"]

    model = {'id':"mixtral-8x7b-32768", "name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}

    max_tokens = model["tokens"]
    for i, prompt in enumerate(seceu_item):
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

        print(f"Item {i}: {ai_responses}")
        try:
            match = re.search(r'\{.*?\}', ai_responses, re.DOTALL)
            if match:
                extracted_dict = eval(match.group(0))
                print("Extracted Dictionary:", extracted_dict)
                res[i] = extracted_dict
            else:
                print("No dictionary found in the response.")
        except Exception as e:
            print(f"Error extracting dictionary: {e}")

        time.sleep(1)

    with open('seceu_res.pkl', 'wb') as f:
        pickle.dump(res, f)
        
if __name__ == "__main__":
    main()

