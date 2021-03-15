import pickle

import pandas as pd
import streamlit as st

import src.pages.home
import src.pages.model_monitor
from src.db.database import RedisDB

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