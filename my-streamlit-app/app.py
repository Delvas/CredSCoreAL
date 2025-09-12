
import streamlit as st
def main():
    st.title("Digital Pay App for Scoring")
    st.write("Welcome to my first Streamlit application!")
    
    # User input
    user_input = st.text_input("Enter some text:")
    
    if st.button("Submit"):
        st.write(f"You entered: {user_input}")

if __name__ == "__main__":
    main()