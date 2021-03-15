import pandas as pd
import pickle
import streamlit as st
from src.db.database import RedisDB
import src.pages.model_monitor
import src.pages.home

PAGES = {
    "Home": src.pages.home,
    "Model Monitor": src.pages.model_monitor,
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.write()

if __name__ == '__main__':
    main()