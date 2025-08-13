from src.pipeline.rag_pipeline import create_rag_pipeline
import streamlit as st
import random
import time

def main():
    path = "settings.json"
    
    st.markdown("<h1 style='text-align: center;'>Lazy to peruse my resumé? Chat with \"Isaac\"</h1>", unsafe_allow_html=True)

    rag_pipeline = create_rag_pipeline(config_path=path)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Chat to find out more about Isaac!"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking about it..."):    
            response = rag_pipeline.query(prompt)
            answer = response.get("answer", "")

        with st.chat_message("assistant"):
            st.markdown(f'''Here you go: ''')
            message_placeholder = st.empty()
            full_response = ""

            # animate typing with code block
            for i in range(len(answer)):
                full_response += answer[i]
                message_placeholder.markdown(f"{full_response}")
                time.sleep(random.uniform(0.001, 0.01))

            st.session_state.messages.append(
                {
                    "role":"AI",
                    "content":full_response,
                }
            )

if __name__ == "__main__":
    main()