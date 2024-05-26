import streamlit as st
import seaborn as sns
import plotly.express as px
from backend import montecarlo

with st.sidebar:
    side_select = st.selectbox(
        "Options: ",
        ["Simple Monte Carlo"],
    )
    if side_select == "Simple Monte Carlo":
        results = None
        success_rate = st.slider("Success Rate", 0.0, 1.0, 0.55, 0.01)
        position_size = st.slider("Position Size", 0.0, 1.0, 0.1, 0.01)
        avg_win = st.slider("Average Win", 0.0, 1.0, 0.08, 0.01)
        avg_loss = st.slider("Average Loss", -1.0, 0.0, -0.02, 0.01)
        num_simulations = st.slider("Number of Simulations", 100, 10000, 10000, 100)
        initial_portfolio_value = st.slider(
            "Initial Portfolio Value", 100, 100000, 100, 100
        )
        fee = st.slider("Transaction Fee", 0.0, 1.0, 0.01, 0.01)
        num_trades = st.slider("Number of Trades", 1, 1000, 100, 1)
        if st.button("Run Simulation"):
            results = montecarlo.run_simple_simulation(
                success_rate=success_rate,
                position_size=position_size,
                avg_win=avg_win,
                avg_loss=avg_loss,
                num_simulations=num_simulations,
                initial_portfolio_value=initial_portfolio_value,
                fee=fee,
                num_trades=num_trades,
            )
            st.success("Simulation Completed")

st.header("Results")
if results is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.write("Starting Portfolio Value: ", results["initial_portfolio_value"])
        st.write("Success Rate: ", results["success_rate"])
        st.write("Position Size: ", results["position_size"])
        st.write("Average Win: ", results["avg_win"])
        st.write("Average Loss: ", results["avg_loss"])
        st.write("Number of Simulations: ", results["num_simulations"])
        st.write("Transaction Fee: ", results["fee"])
        st.write("Number of Trades: ", results["num_trades"])
        st.write("Mean Portfolio Value: ", results["mean_portfolio"])
        st.write("Median Portfolio Value: ", results["median_portfolio"])
        st.write(
            "Standard Deviation of Portfolio Value: ", results["std_dev_portfolio"]
        )
        st.write("5th Percentile Portfolio Value: ", results["percentile_5"])
        st.write("95th Percentile Portfolio Value: ", results["percentile_95"])
    with col2:
        st.write("Portfolio Value Distribution: ")
        fig = px.histogram(results["portfolio_values"])
        st.plotly_chart(fig)
