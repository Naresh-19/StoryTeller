import spacy
import requests
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Initialize environment variables (if you have any like API keys)
load_dotenv()

# Initialize the Chat-based LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the story prompt
story_prompt = ChatPromptTemplate.from_template("""
You are a creative storyteller. Continue and complete the following story in an engaging way):
{story_start}
""")

# Define function to extract keywords from story parts for image fetching
def extract_keywords(text, num_keywords=3):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in {"NOUN", "PROPN", "ADJ"}]
    return keywords[:num_keywords]

def fetch_image_from_unsplash(keywords):
    ACCESS_KEY = "KCVu225-14bcGYi3FBKT0e-K6xRKpYGJN1vR2KihUuo"  # Replace with your Unsplash API key
    query = " ".join(keywords)
    url = "https://api.unsplash.com/search/photos"
    params = {"query": query, "per_page": 1}
    headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}

    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['urls']['regular']
        else:
            return None
    else:
        return None

def download_image(url, save_path):
    response = requests.get(url+"story_start")
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)

# Streamlit UI
st.title('Story Generator with Images')

# Input for story start
story_start = st.text_area("Enter the starting of your story", "Once upon a time, there lived a lion")

if st.button('Generate Story'):
    # Create the LangChain chain
    story_chain = story_prompt | llm
    completed_story = story_chain.invoke({"story_start": story_start})

    # Split the story into parts
    story_parts = completed_story.content.split("\n\n")

    # Directory to save images
    output_dir = "story_images"
    os.makedirs(output_dir, exist_ok=True)

    # Display story parts with images
    for idx, part in enumerate(story_parts):
        # st.subheader(f"Story Part {idx + 1}")
        st.write(part)
        
        # Extract keywords and fetch images
        keywords = extract_keywords(part)
        image_url = fetch_image_from_unsplash(keywords)
        
        if image_url:
            st.image(image_url, caption=f"Image for Part {idx + 1}")
        else:
            st.write("No image available for this part.")
