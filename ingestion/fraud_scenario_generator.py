# ingestion/fraud_scenario_generator.py
"""
Fraud Scenario Generator

This module contains functions and classes to inject fraudulent patterns
into synthetic transactions. It provides different fraud scenarios that
can be applied to a base transaction to make it look suspicious.
"""

import random
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class FraudScenarioConfig:
    """Configuration for fraud scenario injection."""
    # Probability of fraud occurrence (0 to 1)
    fraud_probability: float = 0.02
    # Weights for each fraud type (sum to 1)
    scenario_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.scenario_weights is None:
            self.scenario_weights = {
                "amount_anomaly": 0.3,
                "location_anomaly": 0.25,
                "unusual_hour": 0.15,
                "rapid_succession": 0.2,
                "card_not_present": 0.1,
            }


class FraudScenarioGenerator:
    """
    Applies fraud scenarios to transactions.
    """

    def __init__(self, config: FraudScenarioConfig = None):
        self.config = config or FraudScenarioConfig()
        self._last_transaction_time = {}  # customer_id -> last timestamp

    def should_be_fraud(self) -> bool:
        """Determine if a transaction should be marked as fraud based on probability."""
        return random.random() < self.config.fraud_probability

    def apply_fraud_scenario(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a fraud scenario to the transaction, modifying fields as needed.
        Returns the modified transaction with an added 'is_fraud' flag set to True.
        """
        scenario = self._select_scenario()
        # Apply the selected scenario
        if scenario == "amount_anomaly":
            transaction = self._apply_amount_anomaly(transaction)
        elif scenario == "location_anomaly":
            transaction = self._apply_location_anomaly(transaction)
        elif scenario == "unusual_hour":
            transaction = self._apply_unusual_hour(transaction)
        elif scenario == "rapid_succession":
            transaction = self._apply_rapid_succession(transaction)
        elif scenario == "card_not_present":
            transaction = self._apply_card_not_present(transaction)

        # Mark as fraud (ground truth)
        transaction["is_fraud"] = True
        return transaction

    def _select_scenario(self) -> str:
        """Randomly select a fraud scenario based on configured weights."""
        scenarios = list(self.config.scenario_weights.keys())
        weights = list(self.config.scenario_weights.values())
        return random.choices(scenarios, weights=weights, k=1)[0]

    def _apply_amount_anomaly(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set transaction amount to be significantly higher than the customer's
        typical spending (e.g., 5-20 times average).
        """
        # Assuming transaction has 'amount' and maybe 'average_transaction_amount'
        # We'll just multiply by a random factor
        factor = random.uniform(5, 20)
        transaction["amount"] = round(transaction["amount"] * factor, 2)
        # Also change merchant to a high-value category
        transaction["merchant_category"] = random.choice(
            ["electronics", "travel", "jewelry"]
        )
        return transaction

    def _apply_location_anomaly(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set transaction location to be far away from the customer's base location.
        """
        # We'll assume transaction has 'location' dict with latitude/longitude
        # For simplicity, set to a random distant location (just modify coordinates)
        # In real code, would use geospatial distance.
        transaction["location"]["latitude"] = round(
            random.uniform(-90, 90), 6
        )
        transaction["location"]["longitude"] = round(
            random.uniform(-180, 180), 6
        )
        # Also set country to a random one
        transaction["location"]["country"] = self._get_random_country()
        return transaction

    def _apply_unusual_hour(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set transaction timestamp to an unusual hour (e.g., 2-4 AM).
        """
        # We'll assume transaction has 'timestamp' as milliseconds epoch
        # Convert to datetime, adjust hour, convert back
        ts = transaction["timestamp"]
        dt = datetime.fromtimestamp(ts / 1000.0)
        # Keep date, set hour to between 2 and 4 AM
        new_dt = dt.replace(hour=random.randint(2, 4), minute=random.randint(0, 59))
        transaction["timestamp"] = int(new_dt.timestamp() * 1000)
        return transaction

    def _apply_rapid_succession(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate multiple transactions in quick succession from the same customer.
        This might be reflected by timestamp being very close to last transaction.
        """
        customer_id = transaction["customer_id"]
        # If we have a last transaction time for this customer, set timestamp close to it
        if customer_id in self._last_transaction_time:
            last_ts = self._last_transaction_time[customer_id]
            # Set timestamp to within 1-5 minutes after last
            new_ts = last_ts + random.randint(60, 300) * 1000  # milliseconds
            transaction["timestamp"] = new_ts
        else:
            # No history, just set a recent timestamp
            pass
        # Update last transaction time (will be updated by caller)
        return transaction

    def _apply_card_not_present(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mark transaction as card-not-present (online) despite customer usually
        using card-present.
        """
        transaction["card_present"] = False
        # Also set device to a non-typical one
        transaction["device_id"] = "unknown_device"
        transaction["ip_address"] = "203.0.113.1"  # Example IP
        return transaction

    def _get_random_country(self) -> str:
        """Return a random country code/name."""
        # Could use faker but for simplicity
        countries = [
            "United States",
            "Canada",
            "United Kingdom",
            "Germany",
            "France",
            "Australia",
            "Japan",
            "Brazil",
        ]
        return random.choice(countries)

    def update_last_transaction_time(self, customer_id: str, timestamp: int):
        """Update the last transaction time for a customer after generating a transaction."""
        self._last_transaction_time[customer_id] = timestamp