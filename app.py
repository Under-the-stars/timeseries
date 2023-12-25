import datetime
import os

import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage

import gru_inference
import gru_train_infer

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")  # ""#
    #os.environ["OPENAI_API_KEY"] = openai_api_key
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
greeting_msg = "Hello! How Can I help you ? Do you want Predict stocks today?"
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=f"""
        You are a Stock market expert `chat bot` to help the in extracting  date and company name or ticker using followinf conversation instructions.
        Instructions:
            1) Greet the User with only Greeting message or small talk. Do not send additional Information.
            2) When you are asked to predict the stock price. Strictly ask for both company name and date together and date in DD-MM-YYYY format after mention date: only
            3) Do not hallucinate or assume any thing ask for missing data
                """
    ),
    # The persistent system prompt

    MessagesPlaceholder(variable_name="chat_history"),  # Where the memory will be stored.
    HumanMessagePromptTemplate.from_template("{human_input}"),  # Where the human input will injected

])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": greeting_msg}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hi!"):
    # Add user message to chat history

    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    forcast_msg = None
    with st.chat_message("user"):
        st.markdown(prompt)

        if 'date'.lower().strip() in prompt and "app" in prompt.lower().strip():
            date = prompt.split(":")[-1].strip()
            date_format = "%d-%m-%Y"
            ticker = "AAPL"
            # value = gru_inference.predict_(end=datetime.datetime.strptime(date, date_format),ticker=ticker)
            value = gru_train_infer.train_infer(end=datetime.datetime.strptime(date, date_format), ticker=ticker)
            forcast_msg = f"Stock price of {ticker} on {date} is $ {value}"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if forcast_msg:
            full_response = forcast_msg + " Thank-you!"

        else:
            # Simulate stream of response with milliseconds delay
            full_response = chat_llm_chain.predict(human_input=prompt)

        # print(prompt)
        if "apple" in full_response:
            ticker = 'AAPL'
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # st.session_state.clicked = True
            print("apple")
            st.rerun()
        elif "date" in full_response.lower():
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # st.session_state.file_upload_flag = True
            print("date")
            st.rerun()

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

#
# chat_llm=LLMChain(llm=llm,prompt=prompt,verbose=True,)
# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
# openai.api_key = "sk-"
# st.title("💬 Langchain Powered Chatbot to help you with your Stocks")
#
# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
#
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])
#
# if prompt := st.chat_input():
#     if not openai_api_key:
#         st.info("Please add your OpenAI API key to continue.")
#         st.stop()
#
#     client = openai.OpenAI(api_key=openai_api_key)
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
#
#     response = client.completions.create(model="gpt-3.5-turbo", prompt=prompt)
#     msg = response.choices[0].text
#     st.session_state.messages.append({"role": "assistant", "content": msg})
#     st.chat_message("assistant").write(msg)
#
#     # Extract key features
#     key_features = extract_key_features(st.session_state["messages"])
#
#     # Convert key features to keywords
#     keywords = []
#     for feature in key_features:
#         if feature == "apple":
#             keywords.append("AAPL")
#         elif feature == "microsoft":
#             keywords.append("MSFT")
#
#     # Make a call to a deep learning model
#     prediction = deep_learning_model(keywords)
#
#     # Print the prediction on the frontend
#     st.write(prediction)
#
#
# def extract_key_features(messages):
#     features = []
#     for message in messages:
#         if message["role"] == "user":
#             features.extend(message["content"].split(" "))
#     return list(set(features))
#
#
# def deep_learning_model(keywords):
#     # Replace this with your deep learning model code
#     return 1.234
