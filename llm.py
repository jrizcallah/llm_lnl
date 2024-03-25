import streamlit as st
from chatgpt_connector import get_chatgpt_completion


# App title
st.set_page_config(page_title='JohnnAI')

# Store LLM response
if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant",
                                     "content": "Hi! My name is JohnnAI. "
                                                "John Rizcallah built me to assist with his Lunch and Learn series on LLMs."
                                                "How can I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message['content'])


# Function to generate responses
def generate_response(input_prompt):
    response = get_chatgpt_completion(input_prompt)
    return response


# User-provided prompt
if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append({'role': 'user',
                                      'content': prompt})

    with st.chat_message('user'):
        st.write(prompt)


if st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            response = generate_response(prompt)
            st.write(response)

    message = {'role': 'assistant',
               'content': response}
    st.session_state.messages.append(message)
