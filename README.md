# Taylor-s-Tune
Mood-driven music recommendations with Large Language Models currently running on []()


## Installation

Create a `secrets.toml` in this root folder file containing the following keys: `GROQ_API_KEY`, `QDRANT_API_KEY`, `QDRANT_CLIENT_URL`. Here you can better check out how [secrets work](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

### Option 1: Local installation

1. Copy the `secrets.toml` in the apposite folder ([instructions here](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)) 

2. Install the required packages with 
        
        pip3 install -r requirements.txt

### Option 2: Using Dockerfile

1. Build the app

        docker build -t taylor .
2. Run the app
    
        docker run -p 8501:8501 taylor

