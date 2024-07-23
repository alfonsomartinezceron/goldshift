import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


st.write('Select a commodity to view its historical values')
df=["Gold", "Oil", "Natural Gas", "Silver"]
commodity = st.selectbox(
        "Choose a commodity", df)

#list of excel files
data = {
    "Gold": "/Users/fatoucamarathiam/code/alfonsomartinezceron/goldshift/interface/raw_data/gold.xslx",
    "Oil": "raw_data/oil.csv",
    "Natural Gas": "raw_data/naturalgas.csv"
    }

# Load data based on the selected item
def load_data(file_path):
    return pd.read_excel(file_path)

excel_file = data[commodity]
df = load_data(excel_file)

st.header("Historical data")

#Chart for each selected option
def plot_data(dataframe):
    # plt.figure(figsize=(10, 6))
    # plt.plot(dataframe['DATE'], dataframe['USD'], marker='o')
    # plt.title(f'Plot for {commodity}')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.xticks(rotation=45)
    # plt.grid(True)
    # st.pyplot(plt)

    fig = px.line(dataframe, x='DATE', y='USD', title=f'Interactive Plot for {commodity}')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Value')
    fig.update_layout(title=dict(x=0.5))
    st.plotly_chart(fig)

plot_data(df)
