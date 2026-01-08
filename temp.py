import streamlit as st

st.title("My First Streamlit App")
st.write("Hello, world!")

# Add more interactive components here
st.slider("Select a number", 0, 100, 25)
st.button("Click Me")
