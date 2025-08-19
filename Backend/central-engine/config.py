import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Define multiple topics
KAFKA_TOPICS = {
    "sensor_data": "sensor_topic",
    "accident_data": "accident_topic"
}

KAFKA_GROUPS = {
    "atkins": "atkins_group",
    "atkins-2": "atkins_group_2",
    "atkins-3" : "atkins_group_3",
    "atkins-4": "atkins_group_4"
}