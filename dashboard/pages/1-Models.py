import streamlit as st
import pandas as pd
from backend import inference
from backend import datacheck
import plotly.graph_objects as go
import time

with st.sidebar:
    select_model_type = st.selectbox("Options: ", ["Benchmark", "Champion"])
    st.divider()
    if select_model_type == "Benchmark":
        st.write("Benchmark")
        select_model_ = st.selectbox(
            "Available Models:", inference.all_models(select_model_type.lower())
        )
    st.divider()
    select_signaling_function = st.selectbox("Signaling Function", ["simple"])

st.header("Live inference")
st.write("Live data will be displayed here")
all_transitions_since_live = []  # Using a list to store dictionaries
chart_placeholder = st.empty()
chart_placeholder_2 = st.empty()
text_placeholder = st.empty()

while True:
    result = datacheck.live_data()
    pred, log_return_df, transition = inference.univariate_LR_inference(
        model_name=select_model_, model_version=1
    )
    log_return_df = log_return_df.tail(20)

    if not result.empty:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=result["Opentime"],
                    open=result["Open"],
                    high=result["High"],
                    low=result["Low"],
                    close=result["Close"],
                )
            ]
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    if not log_return_df.empty:
        for _, row in transition.iterrows():
            transition_dict = {"index": _, "log_return": row["log_return"]}
            all_transitions_since_live.append(transition_dict)  # Append each transition

        temp_df = pd.DataFrame(all_transitions_since_live)
        fig2 = go.Figure(
            data=[
                go.Scatter(
                    x=log_return_df.index,
                    y=log_return_df["log_return"],
                    mode="lines",
                    name="log_return",
                ),
                go.Scatter(
                    x=temp_df["index"],
                    y=temp_df["log_return"],
                    mode="lines+markers",
                    name="Prediction",
                    marker=dict(size=10, color="red"),
                ),
            ]
        )
        fig2.update_layout(
            xaxis_title="Time",
            yaxis_title="Log Return",
            title="Log Return and Prediction Over Time",
            showlegend=False,
        )
        chart_placeholder_2.plotly_chart(fig2, use_container_width=True)
    time.sleep(20)
