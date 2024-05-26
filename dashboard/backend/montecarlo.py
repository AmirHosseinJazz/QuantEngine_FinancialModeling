import pandas as pd
import numpy as np
import datetime
import os
from os import path
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns


def simple_monte_carlo(
    success_rate=0.55,
    position_size=0.1,
    avg_win=0.08,
    avg_loss=-0.02,
    num_simulations=10000,
    initial_portfolio_value=100,
    fee=0.01,
    num_trades=100,
):
    """
    This function calculates the portfolio value for each simulation and returns the results.
    param success_rate: percentage of successful trades
    param position_size: position size in percentage
    param avg_win: average win amount in percentage
    param avg_loss: average loss amount in percentage
    param num_simulations: number of simulations
    param initial_portfolio_value: initial portfolio value in dollars
    param fee: transaction fee per trade in percentage
    return: portfolio_values

    """
    position = position_size * initial_portfolio_value  # position size in dollars

    # Set the ranges and distributions of the variables
    success_rate = np.random.normal(success_rate, 0.05, num_simulations)
    win_amount = np.random.normal(avg_win * position, position * 0.01, num_simulations)
    loss_amount = np.random.normal(
        avg_loss * position, position * 0.01, num_simulations
    )

    # Calculate the portfolio value for each simulation
    portfolio_values = np.zeros(num_simulations) + initial_portfolio_value
    for i in range(num_simulations):
        for j in range(num_trades):
            if np.random.random() < success_rate[i]:
                profit = position * win_amount[i]
                transaction = position * fee
                portfolio_values[i] += profit - transaction
            else:
                loss = position * loss_amount[i]
                transaction = position * fee
                portfolio_values[i] += loss - transaction
            if portfolio_values[i] < 0:
                portfolio_values[i] = 0

    # Analyze the results
    mean_portfolio = np.mean(portfolio_values)
    median_portfolio = np.median(portfolio_values)
    std_dev_portfolio = np.std(portfolio_values)
    percentile_5 = np.percentile(portfolio_values, 5)
    percentile_95 = np.percentile(portfolio_values, 95)
    dict_result = {}
    dict_result["mean_portfolio"] = mean_portfolio
    dict_result["median_portfolio"] = median_portfolio
    dict_result["std_dev_portfolio"] = std_dev_portfolio
    dict_result["percentile_5"] = percentile_5
    dict_result["percentile_95"] = percentile_95
    dict_result["portfolio_values"] = portfolio_values
    return dict_result


def run_simple_simulation(
    success_rate=0.55,
    position_size=0.1,
    avg_win=0.08,
    avg_loss=-0.02,
    num_simulations=10000,
    initial_portfolio_value=100,
    fee=0.01,
    num_trades=100,
):
    results = simple_monte_carlo(
        success_rate=success_rate,
        position_size=position_size,
        avg_win=avg_win,
        avg_loss=avg_loss,
        num_simulations=num_simulations,
        initial_portfolio_value=initial_portfolio_value,
        fee=fee,
        num_trades=num_trades,
    )
    results["success_rate"] = success_rate
    results["position_size"] = position_size
    results["avg_win"] = avg_win
    results["avg_loss"] = avg_loss
    results["num_simulations"] = num_simulations
    results["initial_portfolio_value"] = initial_portfolio_value
    results["fee"] = fee
    results["num_trades"] = num_trades
    return results
