# ingestion/transaction_generator.py
"""
Transaction Generator for Fraud Detection Platform

Generates synthetic financial transactions by combining entity data
and optionally injecting fraud scenarios, then optionally produces them
to a Kafka topic.

The generator can be run standalone to produce a stream of transactions,
or used as a library to generate individual transactions for testing.
"""

import json
import logging
import random
import time
import uuid
from typing import Dict, Any, Optional, Generator
from datetime import datetime, timezone
from confluent_kafka import Producer

from entity_generator import EntityGenerator
from fraud_scenario_generator import FraudScenarioGenerator, FraudScenarioConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TransactionGenerator:
    """
    Generates synthetic transactions using entity pools and fraud scenarios.
    Can either yield transactions (for testing) or produce to Kafka.
    """

    def __init__(
        self,
        entity_generator: Optional[EntityGenerator] = None,
        fraud_scenario_generator: Optional[FraudScenarioGenerator] = None,
        fraud_probability: float = 0.02,
        seed: Optional[int] = None,
    ):
        """
        Initialize the transaction generator.

        Args:
            entity_generator: An instance of EntityGenerator, or None to create default.
            fraud_scenario_generator: An instance of FraudScenarioGenerator, or None to create default.
            fraud_probability: Probability of a transaction being fraudulent (0-1).
            seed: Random seed for reproducibility.
        """
        if seed:
            random.seed(seed)

        self.entity_generator = entity_generator or EntityGenerator()
        self.fraud_scenario_generator = fraud_scenario_generator or FraudScenarioGenerator(
            FraudScenarioConfig(fraud_probability=fraud_probability)
        )
        self.fraud_probability = fraud_probability

    def generate_transaction(self) -> Dict[str, Any]:
        """
        Generate a single transaction dictionary.

        Returns:
            Dict containing transaction fields matching the Avro schema.
        """
        # Select customer
        customer = self.entity_generator.get_random_customer()

        # Determine if this transaction should be fraudulent
        is_fraud = self.fraud_scenario_generator.should_be_fraud()

        # Select merchant
        if is_fraud:
            # For fraud, sometimes use a non-typical merchant
            if random.random() < 0.6:
                merchant = self.entity_generator.get_random_merchant_not_typical(customer)
            else:
                merchant = self.entity_generator.get_random_merchant(customer)
        else:
            merchant = self.entity_generator.get_random_merchant(customer)

        # Select device
        device = self.entity_generator.get_random_device(customer)

        # Generate base transaction
        now = datetime.now(timezone.utc)
        timestamp_ms = int(now.timestamp() * 1000)

        # Base location (customer's typical area)
        if random.random() < 0.7:  # 70% transactions near customer's base
            location = {
                "latitude": round(customer.base_location[0] + random.uniform(-0.5, 0.5), 6),
                "longitude": round(customer.base_location[1] + random.uniform(-0.5, 0.5), 6),
                "country": customer.typical_countries[0],
                "city": self._get_city_for_country(customer.typical_countries[0]),
            }
        else:
            # Some random location
            location = {
                "latitude": round(random.uniform(-90, 90), 6),
                "longitude": round(random.uniform(-180, 180), 6),
                "country": self._get_random_country(),
                "city": "Unknown",
            }

        # Amount based on merchant average and customer average
        base_amount = min(
            merchant.average_transaction_amount,
            customer.average_transaction_amount * 2,
        )
        amount = round(abs(random.gauss(base_amount, base_amount * 0.3)), 2)

        # Transaction type
        if merchant.category == "fuel":
            transaction_type = "purchase"
        elif merchant.category == "travel":
            transaction_type = random.choice(["purchase", "booking"])
        else:
            transaction_type = random.choice(
                ["purchase", "withdrawal", "transfer"]
            )

        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "customer_id": customer.customer_id,
            "merchant_id": merchant.merchant_id,
            "merchant_category": merchant.category,
            "amount": amount,
            "currency": "USD",
            "timestamp": timestamp_ms,
            "location": location,
            "device_id": device.device_id,
            "ip_address": device.ip_address,
            "card_present": random.random() < 0.7,  # 70% card present
            "card_type": random.choice(["credit", "debit", "prepaid"]),
            "account_age_days": customer.account_age_days,
            "transaction_type": transaction_type,
            "is_fraud": None,  # Will be set if fraud
        }

        # Apply fraud scenario if needed
        if is_fraud:
            transaction = self.fraud_scenario_generator.apply_fraud_scenario(
                transaction
            )
            # Ensure is_fraud flag is set (already done by apply_fraud_scenario)
            # But we set it again for clarity
            transaction["is_fraud"] = True
        else:
            # Non-fraud: also set is_fraud to False (or leave None)
            # For training, we might want explicit False
            transaction["is_fraud"] = False

        # Update last transaction time for rapid succession scenario
        self.fraud_scenario_generator.update_last_transaction_time(
            transaction["customer_id"], transaction["timestamp"]
        )

        return transaction

    def generate_stream(
        self,
        num_transactions: Optional[int] = None,
        delay_ms: float = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a stream of transactions.

        Args:
            num_transactions: Number of transactions to generate; if None, infinite.
            delay_ms: Delay between transactions in milliseconds (0 for no delay).

        Yields:
            Transaction dictionaries.
        """
        count = 0
        while num_transactions is None or count < num_transactions:
            yield self.generate_transaction()
            count += 1
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    def produce_to_kafka(
        self,
        bootstrap_servers: str,
        topic: str,
        num_transactions: Optional[int] = None,
        delay_ms: float = 100,
        security_protocol: str = "PLAINTEXT",
        sasl_mechanism: Optional[str] = None,
        sasl_username: Optional[str] = None,
        sasl_password: Optional[str] = None,
        ssl_ca_location: Optional[str] = None,
    ):
        """
        Produce generated transactions to a Kafka topic.

        Args:
            bootstrap_servers: Kafka bootstrap servers (comma-separated).
            topic: Kafka topic to produce to.
            num_transactions: Number of transactions to generate; None for infinite.
            delay_ms: Delay between messages in milliseconds.
            security_protocol: Kafka security protocol (PLAINTEXT, SASL_SSL, etc.)
            sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, etc.)
            sasl_username: SASL username.
            sasl_password: SASL password.
            ssl_ca_location: Path to CA certificate for SSL.
        """
        # Configure Kafka producer
        producer_config = {
            "bootstrap.servers": bootstrap_servers,
            "security.protocol": security_protocol,
        }
        if sasl_mechanism:
            producer_config["sasl.mechanism"] = sasl_mechanism
            producer_config["sasl.username"] = sasl_username
            producer_config["sasl.password"] = sasl_password
        if ssl_ca_location:
            producer_config["ssl.ca.location"] = ssl_ca_location

        producer = Producer(producer_config)

        def delivery_report(err, msg):
            if err is not None:
                logger.error(f"Delivery failed for message: {err}")
            else:
                logger.debug(
                    f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
                )

        logger.info(f"Starting Kafka producer to {bootstrap_servers}, topic={topic}")
        count = 0
        try:
            for transaction in self.generate_stream(
                num_transactions, delay_ms
            ):
                # Serialize to JSON (or Avro if needed)
                message = json.dumps(transaction).encode("utf-8")
                # Use transaction_id as key for partitioning
                key = transaction["transaction_id"].encode("utf-8")
                producer.produce(topic, key=key, value=message, callback=delivery_report)
                # Trigger any available delivery report callbacks
                producer.poll(0)
                count += 1
                if count % 100 == 0:
                    logger.info(f"Produced {count} messages")
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            logger.info(f"Flushing producer... (total messages: {count})")
            producer.flush()
            logger.info("Producer closed")

    # Helper methods
    def _get_city_for_country(self, country: str) -> str:
        """Return a random city for a given country (simplified)."""
        # In a real system, you'd use geocoding or a city database.
        # Here we just return a generic city name.
        cities = {
            "United States": "New York",
            "Canada": "Toronto",
            "United Kingdom": "London",
            "Germany": "Berlin",
            "France": "Paris",
            "Australia": "Sydney",
            "Japan": "Tokyo",
            "Brazil": "Sao Paulo",
        }
        return cities.get(country, "Unknown")

    def _get_random_country(self) -> str:
        """Return a random country name."""
        countries = [
            "United States",
            "Canada",
            "United Kingdom",
            "Germany",
            "France",
            "Australia",
            "Japan",
            "Brazil",
            "India",
            "China",
            "Mexico",
            "Spain",
            "Italy",
            "Netherlands",
            "Sweden",
            "Singapore",
            "United Arab Emirates",
            "South Africa",
        ]
        return random.choice(countries)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic transactions and produce to Kafka"
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:9092",
        help="Kafka bootstrap servers",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="transactions.raw",
        help="Kafka topic",
    )
    parser.add_argument(
        "--num-transactions",
        type=int,
        default=None,
        help="Number of transactions to generate (default: infinite)",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=100,
        help="Delay between transactions in milliseconds",
    )
    parser.add_argument(
        "--fraud-probability",
        type=float,
        default=0.02,
        help="Probability of fraud (0-1)",
    )
    parser.add_argument(
        "--security-protocol",
        type=str,
        default="PLAINTEXT",
        help="Kafka security protocol",
    )
    parser.add_argument(
        "--sasl-mechanism",
        type=str,
        default=None,
        help="SASL mechanism",
    )
    parser.add_argument(
        "--sasl-username",
        type=str,
        default=None,
        help="SASL username",
    )
    parser.add_argument(
        "--sasl-password",
        type=str,
        default=None,
        help="SASL password",
    )

    args = parser.parse_args()

    generator = TransactionGenerator(
        fraud_probability=args.fraud_probability
    )
    generator.produce_to_kafka(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        num_transactions=args.num_transactions,
        delay_ms=args.delay_ms,
        security_protocol=args.security_protocol,
        sasl_mechanism=args.sasl_mechanism,
        sasl_username=args.sasl_username,
        sasl_password=args.sasl_password,
    )