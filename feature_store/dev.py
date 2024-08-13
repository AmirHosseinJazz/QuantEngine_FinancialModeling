from dotenv import load_dotenv
from confluent_kafka import DeserializingConsumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import StringDeserializer


def process_trade(trade):
    global current_volume, current_bar
    price = float(trade["price"])
    volume = float(trade["quantity"])
    current_volume += volume

    # Add trade to the list of trades for the current bar
    current_bar["trades"].append(trade)

    # Set open price if it's the first trade in the bar
    if current_bar["open"] is None:
        current_bar["open"] = price

    # Update high and low prices
    if price > current_bar["high"]:
        current_bar["high"] = price
    if price < current_bar["low"]:
        current_bar["low"] = price

    # Update close price with each trade
    current_bar["close"] = price

    # When volume threshold is reached or exceeded
    if current_volume >= volume_threshold:
        emit_volume_bar()
        # Reset for the next bar
        current_volume = 0
        current_bar = {
            "trades": [],
            "open": None,
            "high": float("-inf"),
            "low": float("inf"),
            "close": None,
        }


def emit_volume_bar():
    # Handle the completed volume bar
    print("Volume bar closed")
    print("Open: {}".format(current_bar["open"]))
    print("High: {}".format(current_bar["high"]))
    print("Low: {}".format(current_bar["low"]))
    print("Close: {}".format(current_bar["close"]))
    

if __name__ == "__main__":
    volume_threshold = 10     # Define your volume threshold
    current_volume = 0
    current_bar = {
        "trades": [],
        "open": None,
        "high": float("-inf"),
        "low": float("inf"),
        "close": None,
    }

    value_schema_str = """{
      "type": "record",
      "name": "AggregateTrade",
      "namespace": "com.example.binance",
      "fields": [
        {"name": "eventType", "type": "string"},
        {"name": "eventTime", "type": "long"},
        {"name": "symbol", "type": "string"},
        {"name": "aggTradeId", "type": "long"},
        {"name": "price", "type": "string"},
        {"name": "quantity", "type": "string"},
        {"name": "firstTradeId", "type": "long"},
        {"name": "lastTradeId", "type": "long"},
        {"name": "tradeTime", "type": "long"},
        {"name": "isBuyerMaker", "type": "boolean"},
        {"name": "ignore", "type": "boolean"}
      ]
    }"""

    schema_registry_conf = {"url": "http://schema-registry:8081"}
    schema_registry_client = SchemaRegistryClient(schema_registry_conf)

    # Avro deserializer
    # You should know the schema ID or have the actual schema string for the AvroDeserializer
    # If you have the schema string, replace `None` with the actual schema string
    avro_deserializer = AvroDeserializer(
        schema_str=value_schema_str, schema_registry_client=schema_registry_client
    )

    # Consumer configuration
    consumer_conf = {
        "bootstrap.servers": "kafka:9092",
        "group.id": "test",
        "auto.offset.reset": "earliest",
        "key.deserializer": StringDeserializer("utf_8"),
        "value.deserializer": avro_deserializer,
    }

    consumer = DeserializingConsumer(consumer_conf)
    consumer.subscribe(["btcaggtrade"])
    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                print("No message received")
                continue
                # raise Exception("No message received")
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    raise Exception("End of partition")
                else:
                    print(msg.error())
                    break

            # print("Received message: {}".format(msg.value()))
            process_trade(msg.value())
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
