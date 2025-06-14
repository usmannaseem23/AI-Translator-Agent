import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import asyncio
import os

# Load .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Validate key
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Configure external client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Translator agent
writer = Agent(
    name="Translator Agent",
    instructions="Translate the given English word, name, or sentence into Urdu only. Do not return any explanation or English text—just the Urdu translation."
)


# Async-safe runner
def run_async_safely(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Streamlit UI
st.set_page_config(page_title="English to Urdu Translator", layout="centered")

st.title("🤖AI Translator Agent (Powered By GEMINI)")
st.write("Type any English Text and Sentence and click Translate to see it in Urdu.")

# Input
input_text = st.text_area("✍️ Enter your Text :", height=100)

# Translate button
if st.button("🔁 Translate"):
    if input_text.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            response = run_async_safely(
                Runner.run(writer, input=input_text, run_config=config)
            )
            st.success("Translation:")
            st.markdown(f"📘 **Urdu:** {response.final_output}")  # or response.final_output

