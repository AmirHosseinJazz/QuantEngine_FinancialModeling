import os
import pandas as pd
import ta
from preprocess import get_backtrader_data
from dotenv import load_dotenv
import psycopg2
import json

load_dotenv()


def insert_strategy(
    asset, timeframe, eval_period, period, success_rate, extra_params, strategy_name
):
    conn = psycopg2.connect(
        host=os.getenv("TIMESCALE_HOST_NAME"),
        port=os.getenv("LOCAL_TIMESCALE_PORT"),
        user=os.getenv("TIMESCALE_USER"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DATABASE"),
    )
    cursor = conn.cursor()

    query = f"""
        
    INSERT INTO "BTCUSDT".traditional_strategy_stats(
	asset, timeframe, eval_period, period, success_rate, extra_params, strategy)
    VALUES ('{asset}', '{timeframe}', '{eval_period}', '{period}', {success_rate}, '{json.dumps(extra_params)}', '{strategy_name}')
          ON CONFLICT (asset, timeframe, period, eval_period, strategy)
        DO UPDATE SET
            success_rate = EXCLUDED.success_rate,
            extra_params = EXCLUDED.extra_params;
             """
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
    print("Strategy stats inserted successfully")


def check_MACD(
    asset="BTCUSDT",
    interval="1d",
    window_slow=26,
    window_fast=12,
    window_sign=9,
    candle="Close",
    eval_period=20,
    time_frame="last_year",
):
    # get now unix
    if time_frame == "last_week":
        now = pd.Timestamp.now()
        now_unix = now.value // 10**6
        last_week = now - pd.Timedelta(days=7)
        last_week_unix = last_week.value // 10**6
        data = get_backtrader_data(asset, str(last_week_unix), str(now_unix), interval)

    elif time_frame == "last_month":
        now = pd.Timestamp.now()
        now_unix = now.value // 10**6
        last_month = now - pd.Timedelta(days=30)
        last_month_unix = last_month.value // 10**6
        data = get_backtrader_data(asset, str(last_month_unix), str(now_unix), interval)

    elif time_frame == "last_year":
        now = pd.Timestamp.now()
        now_unix = now.value // 10**6
        last_year = now - pd.Timedelta(days=365)
        last_year_unix = last_year.value // 10**6
        data = get_backtrader_data(asset, str(last_year_unix), str(now_unix), interval)

    elif time_frame == "all":
        data = get_backtrader_data(asset, "1500000000000", "now", interval)

    macd = ta.trend.MACD(
        data[candle],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
    )
    data["MACD"] = macd.macd()
    data["Signal"] = macd.macd_signal()

    # Identify crossovers
    data["Crossover"] = 0
    data.loc[
        (data["MACD"] > data["Signal"])
        & (data["MACD"].shift(1) <= data["Signal"].shift(1)),
        "Crossover",
    ] = 1  # Bullish
    data.loc[
        (data["MACD"] < data["Signal"])
        & (data["MACD"].shift(1) >= data["Signal"].shift(1)),
        "Crossover",
    ] = -1  # Bearish

    # Evaluate performance over the next n candles
    n = eval_period
    results = []

    for index, row in data.iterrows():
        if row["Crossover"] != 0:
            entry_price = row["Close"]
            future_prices = data.loc[index:][:n]["Close"]
            if row["Crossover"] == 1:
                success = any(future_prices > entry_price)
            elif row["Crossover"] == -1:
                success = any(future_prices < entry_price)
            results.append(
                {"Date": index, "Crossover": row["Crossover"], "Success": success}
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate success rate
    success_rate = round(results_df["Success"].mean() * 100, 2)
    return {
        "asset": asset,
        "timeframe": interval,
        "eval_period": eval_period,
        "period": time_frame,
        "success_rate": success_rate,
        "extra_params": {
            "window_slow": window_slow,
            "window_fast": window_fast,
            "window_sign": window_sign,
        },
        "strategy_name": "MACD",
    }


def detect_gaps(data, gap_threshold):
    gaps = []
    for i in range(1, len(data)):
        prev_high = data.iloc[i - 1]["High"]
        curr_low = data.iloc[i]["Low"]
        prev_low = data.iloc[i - 1]["Low"]
        curr_high = data.iloc[i]["High"]

        # Detect a gap up
        # print(curr_low/prev_high)
        if curr_high > (prev_high * (1 + gap_threshold)):
            gaps.append(
                {
                    "Type": "Gap Up",
                    "Gap_Start": prev_high,
                    "Gap_End": curr_high,
                    "Gap_Filled": False,
                    "Fill_Level": prev_high,
                    "Detection_Date": data.index[i],
                    "Fill_Date": None,
                }
            )

        # Detect a gap down
        if curr_low < (prev_low * (1 - gap_threshold)):
            gaps.append(
                {
                    "Type": "Gap Down",
                    "Gap_Start": curr_low,
                    "Gap_End": prev_low,
                    "Gap_Filled": False,
                    "Fill_Level": prev_low,
                    "Detection_Date": data.index[i],
                    "Fill_Date": None,
                }
            )
    return gaps


def monitor_gaps(data, gaps, N):
    for gap in gaps:
        if not gap["Gap_Filled"]:
            detection_index = data.index.get_loc(gap["Detection_Date"])
            for i in range(
                detection_index + 1, min(detection_index + 1 + N, len(data))
            ):
                if gap["Type"] == "Gap Up" and data.iloc[i]["Low"] <= gap["Fill_Level"]:
                    gap["Gap_Filled"] = True
                    gap["Fill_Date"] = data.index[i]
                    break
                elif (
                    gap["Type"] == "Gap Down"
                    and data.iloc[i]["High"] >= gap["Fill_Level"]
                ):
                    gap["Gap_Filled"] = True
                    gap["Fill_Date"] = data.index[i]
                    break
    return gaps


def check_FVG(
    asset="BTCUSDT",
    interval="1d",
    candle="Close",
    eval_period=5,
    time_frame="last_year",
    gap_threshold=0.02,
):
    # get now unix
    if time_frame == "last_week":
        now = pd.Timestamp.now()
        now_unix = now.value // 10**6
        last_week = now - pd.Timedelta(days=7)
        last_week_unix = last_week.value // 10**6
        data = get_backtrader_data(asset, str(last_week_unix), str(now_unix), interval)
    elif time_frame == "last_month":
        now = pd.Timestamp.now()
        now_unix = now.value // 10**6
        last_month = now - pd.Timedelta(days=30)
        last_month_unix = last_month.value // 10**6
        data = get_backtrader_data(asset, str(last_month_unix), str(now_unix), interval)
    elif time_frame == "last_year":
        now = pd.Timestamp.now()
        now_unix = now.value // 10**6
        last_year = now - pd.Timedelta(days=365)
        last_year_unix = last_year.value // 10**6
        data = get_backtrader_data(asset, str(last_year_unix), str(now_unix), interval)
    elif time_frame == "all":
        data = get_backtrader_data(asset, "1500000000000", "now", interval)

    N = eval_period  # Number of periods to check for gap filling

    # Detect gaps in the data
    gaps = detect_gaps(data, gap_threshold)
    # Monitor gaps for filling within N periods
    gaps = monitor_gaps(data, gaps, N)
    # Convert gaps to a DataFrame for analysis
    gaps_df = pd.DataFrame(gaps)

    # Calculate success rate
    success_rate = gaps_df["Gap_Filled"].mean() * 100
    return {
        "asset": asset,
        "timeframe": interval,
        "eval_period": eval_period,
        "period": time_frame,
        "success_rate": success_rate,
        "extra_params": {
            "gap_treshold": gap_threshold,
            "Number of Gaps": len(gaps),
        },
        "strategy_name": "FVG",
    }


def identify_pinbars(df, k=0.2):
    pinbars = []
    for i in range(len(df)):
        O = df.iloc[i]["Open"]
        C = df.iloc[i]["Close"]
        H = df.iloc[i]["High"]
        L = df.iloc[i]["Low"]

        body_size = abs(C - O)
        candle_length = H - L
        lower_shadow = O - L if O < C else C - L
        upper_shadow = H - O if O < C else H - C

        # Bullish Pinbar
        if (
            body_size < k * candle_length
            and lower_shadow >= 2 * body_size
            and upper_shadow <= (1 / 3) * candle_length
        ):
            pinbars.append((df.index[i], "Bullish Pinbar"))

        # Bearish Pinbar
        if (
            body_size < k * candle_length
            and upper_shadow >= 2 * body_size
            and lower_shadow <= (1 / 3) * candle_length
        ):
            pinbars.append((df.index[i], "Bearish Pinbar"))

    return pd.DataFrame(pinbars, columns=["Date", "Pattern"])


def check_trend_reversal(df, pinbars, look_ahead=10, threshold=0.02):
    reversals = []
    for index, row in pinbars.iterrows():
        pinbar_date = row["Date"]
        pattern = row["Pattern"]
        pinbar_index = df.index.get_loc(pinbar_date)

        if pattern == "Bullish Pinbar":
            for i in range(1, look_ahead + 1):
                if pinbar_index + i < len(df):
                    future_price = df.iloc[pinbar_index + i]["Close"]
                    if future_price >= df.iloc[pinbar_index]["Close"] * (1 + threshold):
                        reversals.append(
                            (
                                pinbar_date,
                                pattern,
                                df.index[pinbar_index + i],
                                "Reversal Confirmed",
                            )
                        )
                        break

        elif pattern == "Bearish Pinbar":
            for i in range(1, look_ahead + 1):
                if pinbar_index + i < len(df):
                    future_price = df.iloc[pinbar_index + i]["Close"]
                    if future_price <= df.iloc[pinbar_index]["Close"] * (1 - threshold):
                        reversals.append(
                            (
                                pinbar_date,
                                pattern,
                                df.index[pinbar_index + i],
                                "Reversal Confirmed",
                            )
                        )
                        break
            

    return pd.DataFrame(
        reversals, columns=["Pinbar Date", "Pattern", "Reversal Date", "Reversal"]
    )


def pinbar_strategy(
    asset="BTCUSDT",
    interval="1d",
    eval_period=5,
    time_frame="last_year",
    k=0.2,
    threshold=0.02,
):
    if time_frame == "last_week":
        now = pd.Timestamp.now()
        now_unix = now.value // 10**6
        last_week = now - pd.Timedelta(days=7)
        last_week_unix = last_week.value // 10**6
        data = get_backtrader_data(asset, str(last_week_unix), str(now_unix), interval)
    elif time_frame == "last_month":
        now = pd.Timestamp.now()
        now_unix = now.value // 10**6
        last_month = now - pd.Timedelta(days=30)
        last_month_unix = last_month.value // 10**6
        data = get_backtrader_data(asset, str(last_month_unix), str(now_unix), interval)
    elif time_frame == "last_year":
        now = pd.Timestamp.now()
        now_unix = now.value // 10**6
        last_year = now - pd.Timedelta(days=365)
        last_year_unix = last_year.value // 10**6
        data = get_backtrader_data(asset, str(last_year_unix), str(now_unix), interval)
    elif time_frame == "all":
        data = get_backtrader_data(asset, "1500000000000", "now", interval)


    pinbars = identify_pinbars(data, k)
    reversals = check_trend_reversal(data, pinbars, eval_period, threshold)
    print(len(pinbars))
    print(len(reversals[reversals['Reversal'] == 'Reversal Confirmed']))

if __name__ == "__main__":
    # print("Running MACD")
    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1h",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="last_week",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1h",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="last_month",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1h",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="last_year",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1h",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="all",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )
    # ########### 1d

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1d",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="last_year",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1d",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="last_month",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1d",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="last_week",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1d",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="all",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # ########### 1m
    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1m",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="last_year",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1m",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="last_month",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1m",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="last_week",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # result = check_MACD(
    #     asset="BTCUSDT",
    #     interval="1m",
    #     window_slow=26,
    #     window_fast=12,
    #     window_sign=9,
    #     candle="Close",
    #     eval_period=5,
    #     time_frame="all",
    # )
    # insert_strategy(
    #     asset=result["asset"],
    #     timeframe=result["timeframe"],
    #     eval_period=result["eval_period"],
    #     period=result["period"],
    #     success_rate=result["success_rate"],
    #     extra_params=result["extra_params"],
    #     strategy_name=result["strategy_name"],
    # )

    # print("Running Fair Value GAP (FVG)")

    # res = check_FVG(asset="BTCUSDT", interval="1h", time_frame="last_week")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1h", time_frame="last_month")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1h", time_frame="last_year")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1h", time_frame="all")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1d", time_frame="last_year")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1d", time_frame="last_month")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1d", time_frame="last_week")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1d", time_frame="all")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1m", time_frame="last_year")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1m", time_frame="last_month")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1m", time_frame="last_week")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    # res = check_FVG(asset="BTCUSDT", interval="1m", time_frame="all")
    # insert_strategy(
    #     asset=res["asset"],
    #     timeframe=res["timeframe"],
    #     eval_period=res["eval_period"],
    #     period=res["period"],
    #     success_rate=res["success_rate"],
    #     extra_params=res["extra_params"],
    #     strategy_name=res["strategy_name"],
    # )

    print("Candlestick Pinbar")
    res=pinbar_strategy(
        asset="BTCUSDT",
        interval="1h",
        eval_period=10,
        time_frame="last_year",
        k=0.2,
        threshold=0.005,
    )