import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd


def forex_check(schema_name="EURUSD"):

    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = '{schema_name}'
            """
    cursor.execute(query)

    tables = cursor.fetchall()

    cursor.close()
    conn.close()

    available_tables = []
    if "kline_1M" in [table[0] for table in tables]:
        available_tables.append("One Minute")
    if "kline_5M" in [table[0] for table in tables]:
        available_tables.append("Five Minutes")
    if "kline_15M" in [table[0] for table in tables]:
        available_tables.append("Fifteen Minutes")
    if "kline_30M" in [table[0] for table in tables]:
        available_tables.append("Thirty Minutes")
    if "kline_1H" in [table[0] for table in tables]:
        available_tables.append("One Hour")
    if "kline_4H" in [table[0] for table in tables]:
        available_tables.append("Four Hours")
    if "kline_1D" in [table[0] for table in tables]:
        available_tables.append("One Day")
    return available_tables


def number_of_rows(schema_name="EURUSD", timeframe="kline_1M"):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
            SELECT COUNT(*) FROM "{schema_name}"."{timeframe}"
            """
    cursor.execute(query)

    count = cursor.fetchall()

    cursor.close()
    conn.close()

    return count[0][0]


def number_of_rows_technical(schema_name="BTCUSDT", timeframe="technical_1M"):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
            SELECT COUNT(*) FROM "{schema_name}"."{timeframe}"
            """
    cursor.execute(query)

    count = cursor.fetchall()

    cursor.close()
    conn.close()

    return count[0][0]


def number_of_rows_crypto_events():
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
                SELECT 
                COUNT(*) AS total_rows,
                MAX(date) AS max_date,
                MIN(date) AS min_date
            FROM 
                "news"."crypto_events";
            """
    cursor.execute(query)

    count = cursor.fetchall()

    cursor.close()
    conn.close()

    return count[0]


def number_of_rows_onchain(symbol="BTCUSDT"):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
            SELECT COUNT(*) FROM "{symbol}"."onchain_indicators"
            """
    cursor.execute(query)

    count = cursor.fetchall()

    cursor.close()
    conn.close()

    return count[0][0]


def number_of_rows_onchain_items(symbol="BTCUSDT"):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
            SELECT "Item",COUNT(*) FROM "{symbol}".onchain_indicators GROUP BY "Item"
            """
    cursor.execute(query)

    count = cursor.fetchall()

    cursor.close()
    conn.close()

    return count


def last_update_candles(symbol="BTCUSDT"):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
            select "lastUpdate_1M","lastUpdate_1H","lastUpdate_1D" from 
            (select * from 
            (select 1 as "id1", "Opentime" as "lastUpdate_1M" from "{symbol}"."kline_1M" order by "Opentime" desc limit 1) as t
            inner join 
            (select 1 as "id2", "Opentime" as "lastUpdate_1H" from "{symbol}"."kline_1H" order by "Opentime" desc limit 1) as t2
            on t.id1=t2.id2) as t3
            inner join 
            (select 1 as "id", "Opentime" as "lastUpdate_1D" from "{symbol}"."kline_1D" order by "Opentime" desc limit 1) as t4
            on t3.id1=t4.id
            """
    cursor.execute(query)
    cols = [desc[0] for desc in cursor.description]
    count = cursor.fetchall()

    cursor.close()
    conn.close()
    try:
        return pd.DataFrame(count, columns=cols)
    except:
        return None


def last_update_onchain(symbol="BTCUSDT"):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
            select Distinct on ("Item") "Item","DateTime"  from "{symbol}"."onchain_indicators" order by "Item","DateTime" DESC
            """
    cursor.execute(query)
    cols = [desc[0] for desc in cursor.description]
    count = cursor.fetchall()

    cursor.close()
    conn.close()
    try:
        return pd.DataFrame(count, columns=cols)
    except:
        return None


def last_update_technicals(symbol="BTCUSDT"):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
            select "lastUpdate_1M","lastUpdate_1H","lastUpdate_1D" from 
            (select * from 
            (select 1 as "id1", "CloseTime" as "lastUpdate_1M" from "{symbol}"."technical_1M" order by "CloseTime" desc limit 1) as t
            inner join 
            (select 1 as "id2", "CloseTime" as "lastUpdate_1H" from "{symbol}"."technical_1H" order by "CloseTime" desc limit 1) as t2
            on t.id1=t2.id2) as t3
            inner join 
            (select 1 as "id", "CloseTime" as "lastUpdate_1D" from "{symbol}"."technical_1D" order by "CloseTime" desc limit 1) as t4
            on t3.id1=t4.id
            """
    cursor.execute(query)
    cols = [desc[0] for desc in cursor.description]
    count = cursor.fetchall()

    cursor.close()
    conn.close()
    try:
        return pd.DataFrame(count, columns=cols)
    except:
        return None


def last_update_events():
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()

    query = f"""
            select "date" from "news"."crypto_events" order by "date" desc limit 1
            """
    cursor.execute(query)
    cols = [desc[0] for desc in cursor.description]
    count = cursor.fetchall()

    cursor.close()
    conn.close()
    try:
        return pd.DataFrame(count, columns=cols)
    except:
        return None


def live_data(symbol="BTCUSDT", interval="1m"):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    if interval.lower() == "1m":
        interval = "1M"
    elif interval.lower() == "1h":
        interval = "1H"
    elif interval.lower() == "1d":
        interval = "1D"
    try:
        with psycopg2.connect(
            host=host, database=database, user=user, password=password
        ) as conn:
            with conn.cursor() as cursor:
                query = f"""
                        SELECT * FROM "{symbol}"."kline_{interval}" ORDER BY "Opentime" DESC LIMIT 50
                        """
                cursor.execute(query)
                cols = [desc[0] for desc in cursor.description]
                count = cursor.fetchall()
                data=pd.DataFrame(count, columns=cols)
                data['Opentime'] = pd.to_datetime(data['Opentime'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('CET')
                return data
    except (psycopg2.Error, psycopg2.DatabaseError) as error:
        print(f"Error connecting to the database: {error}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
