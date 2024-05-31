import streamlit as st
from annotated_text import annotated_text
import os
from dotenv import load_dotenv
from backend import datacheck
import time
from datetime import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pytz import timezone

load_dotenv()
_forex_assets = os.getenv("FOREX_ASSETS").split(",")
_crypto_assets = os.getenv("CRYPTO_ASSETS").split(",")

possible_timeframes = {
    "One Minute": "kline_1M",
    "Five Minutes": "kline_5M",
    "Fifteen Minutes": "kline_15M",
    "Thirty Minutes": "kline_30M",
    "One Hour": "kline_1H",
    "Four Hours": "kline_4H",
    "One Day": "kline_1D",
}
possible_timeframes_tech = {
    "One Minute": "technical_1M",
    "Five Minutes": "technical_5M",
    "Fifteen Minutes": "technical_15M",
    "Thirty Minutes": "technical_30M",
    "One Hour": "technical_1H",
    "Four Hours": "technical_4H",
    "One Day": "technical_1D",
}


st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    asset_class_select = st.selectbox(
        "Asset Class?",
        [
            "Crypto",
            # "Forex",
        ],
    )
    st.divider()
    if asset_class_select == "Forex":
        currency_pair = st.selectbox("Currency Pair?", _forex_assets)
    else:
        currency_pair = st.selectbox("Currency Pair?", _crypto_assets)


st.title("Enigma")
st.subheader("Data availability check")

tab1, tab2, tab3 = st.tabs(["Kline-Technical", "Onchain", "Last Update"])

with tab1:
    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
    with col1:
        st.subheader("Timeframes")
        with st.spinner("Checking data availability..."):
            available_tables = datacheck.forex_check(currency_pair)
        _available_tfs = []
        for _p in list(possible_timeframes.keys()):
            if _p not in available_tables:
                st.error(_p)
            else:
                st.success(_p)
                _available_tfs.append(_p)
    with col2:
        st.subheader("Data Preview: ")
        for _tf in _available_tfs:
            annotated_text(
                (_tf, "Timeframe"),
                (
                    str(
                        datacheck.number_of_rows(
                            currency_pair, possible_timeframes[_tf]
                        )
                    ),
                    "Rows",
                ),
            )

    with col3:
        st.subheader("Technical Indicators")
        for _tf in _available_tfs:
            annotated_text(
                (_tf, "Timeframe"),
                (
                    str(
                        datacheck.number_of_rows_technical(
                            currency_pair, possible_timeframes_tech[_tf]
                        )
                    ),
                    "Rows",
                ),
            )
    with col4:
        st.subheader("Event Data: ")
        annotated_text(("Strat Date:", "Date"))
        if asset_class_select == "Forex":
            st.write("No event data available")
        else:
            annotated_text((str(datacheck.number_of_rows_crypto_events()[2]), ""))
        annotated_text(("End Date:", "Date"))
        if asset_class_select == "Forex":
            st.write("No event data available")
        else:
            annotated_text((str(datacheck.number_of_rows_crypto_events()[1]), ""))
        annotated_text(("Number of rows:", "Rows"))
        if asset_class_select == "Forex":
            st.write("No event data available")
        else:
            annotated_text((str(datacheck.number_of_rows_crypto_events()[0]), ""))
        pass

with tab2:
    (
        col1,
        col2,
        col3,
        col4,
    ) = st.columns([1, 1, 1, 1])
    with col1:
        st.subheader("Total On Chain :")
        annotated_text(
            ("Total rows:", ""),
            (str(datacheck.number_of_rows_onchain(symbol=currency_pair)), "Rows"),
        )
    with col2:
        st.subheader("Item disaggregation :")
        _container = st.container(height=500)
        with _container:
            try:
                for _onchain_item in datacheck.number_of_rows_onchain_items(
                    symbol=currency_pair
                ):
                    annotated_text(
                        (_onchain_item[0], "Item"), (str(_onchain_item[1]), "")
                    )
            except:
                st.write("No data available")

with tab3:
    st.subheader("Last Update Candles")
    ##########################################################################
    _last_update_candles = datacheck.last_update_candles(symbol=currency_pair)
    now = datetime.now()
    try:
        _last_update_candles = datacheck.last_update_candles(symbol=currency_pair)
        # 1M
        last_update_1M = datetime.fromtimestamp(
            int(_last_update_candles["lastUpdate_1M"] / 1000)
        )
        cet = timezone("CET")
        cet_time = last_update_1M.astimezone(cet)
        relative_time = relativedelta(now, last_update_1M)
        annotated_text(
            ("1 Minute", "Date"),
            (str(cet_time), ""),
            (str(relative_time.minutes), "Minutes Behind"),
            # (str(_last_update_candles["lastUpdate_1M"].iloc[0]), "Timestamp"),
        )
        # 1H
        last_update_1H = datetime.fromtimestamp(
            int(_last_update_candles["lastUpdate_1H"] / 1000)
        )
        cet_time_1H = last_update_1H.astimezone(cet)
        relative_time = relativedelta(now, last_update_1H)
        annotated_text(
            ("1 Hour", "Date"),
            (str(cet_time_1H), ""),
            (str(relative_time.hours), "Hours Behind"),
        )

        # 1D
        last_update_1D = datetime.fromtimestamp(
            int(_last_update_candles["lastUpdate_1D"] / 1000)
        )
        relative_time = relativedelta(now, last_update_1D)
        cet_time_1D = last_update_1D.astimezone(cet)
        annotated_text(
            ("1 Day", "Date"),
            (str(cet_time_1D), ""),
            (str(relative_time.days), "Days Behind"),
        )
    except Exception as E:
        # st.write(E)
        st.write("No data available")
    ##########################################################################
    st.subheader("Last Update Technical")
    try:
        _last_update_technicals = datacheck.last_update_technicals(symbol=currency_pair)
        # 1M
        last_update_1M = datetime.fromtimestamp(
            int(_last_update_technicals["lastUpdate_1M"] / 1000)
        )
        relative_time = relativedelta(now, last_update_1M)
        cet_time_1M_tech = last_update_1M.astimezone(cet)
        annotated_text(
            ("1 Minute", "Date"),
            (str(cet_time_1M_tech), ""),
            (str(relative_time.minutes), "Minutes Behind"),
        )
        # 1H
        last_update_1H = datetime.fromtimestamp(
            int(_last_update_technicals["lastUpdate_1H"] / 1000)
        )
        cet_time_1H_tech = last_update_1H.astimezone(cet)
        relative_time = relativedelta(now, last_update_1H)
        annotated_text(
            ("1 Hour", "Date"),
            (str(cet_time_1H_tech), ""),
            (str(relative_time.hours), "Hours Behind"),
        )

        # 1D
        last_update_1D = datetime.fromtimestamp(
            int(_last_update_technicals["lastUpdate_1D"] / 1000)
        )
        relative_time = relativedelta(now, last_update_1D)
        cet_time_1D_tech = last_update_1D.astimezone(cet)
        annotated_text(
            ("1 Day", "Date"),
            (str(cet_time_1D_tech), ""),
            (str(relative_time.days), "Days Behind"),
        )
    except Exception as E:
        # st.write(E)
        st.write("No data available")
    #########################################################################
    st.subheader("Last Update Events")
    try:
        _last_update_events = datacheck.last_update_events()
        last_update_start = datetime.strptime(
            _last_update_events["date"][0], "%Y-%m-%d"
        )
        relative_time = relativedelta(now, last_update_start)
        cet_time_event= last_update_start.astimezone(cet)
        if relative_time.days < 0:
            annotated_text(
                ("Last Event", "Date"),
                (str(last_update_start), ""),
                (str(-1 * relative_time.days), "Days Ahead"),
            )
        else:
            annotated_text(
                ("Last Event", "Date"),
                (str(last_update_start), ""),
                (str(relative_time.days), "Days Behind"),
            )
    except Exception as E:
        st.write(E)
        st.write("No data available")

    ##########################################################################
    st.subheader("Last Update Onchain")
    _container = st.container(height=300)
    with _container:
        try:
            _last_update_onchain = datacheck.last_update_onchain(symbol=currency_pair)
            for _, row in _last_update_onchain.iterrows():
                last_update = datetime.strptime(row["DateTime"], "%Y-%m-%d")
                relative_time = relativedelta(now, last_update)
                annotated_text(
                    (row["Item"], "Item"),
                    (str(last_update), ""),
                    (str(relative_time.days), "Days Ago"),
                )

        except Exception as E:
            st.write(E)
            st.write("No data available")
