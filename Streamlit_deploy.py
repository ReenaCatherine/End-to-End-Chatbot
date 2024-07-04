import streamlit as st
from Chatbot import chatbot

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome! Please type your message and press enter")

    counter += 1
    user_input = st.text_input("You:", key = f"user_input_{counter}")

    if user_input:
        response, confidence_score = chatbot(user_input)
        st.text_area("Chatbot:", value = response, height = 100,
                     max_chars = None, key = f"chatbot_response_{counter}")
        #confidence score
        st.write(f"Confidence:{confidence_score:.2f}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()