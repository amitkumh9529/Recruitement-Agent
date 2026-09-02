# ingestion/entity_generator.py
"""
Entity Generator for Fraud Detection Platform

This module generates realistic entities (customers, merchants, devices, locations)
used to create synthetic financial transactions.

It uses the Faker library to generate realistic data and maintains in-memory
pools of entities for consistency across generated transactions.
"""

import random
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from faker import Faker


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    account_age_days: int
    credit_score: int
    typical_countries: List[str]
    typical_categories: List[str]
    typical_transaction_types: List[str]
    base_location: Tuple[float, float]  # (latitude, longitude)
    average_transaction_amount: float


@dataclass
class Merchant:
    merchant_id: str
    name: str
    category: str
    country: str
    city: str
    location: Tuple[float, float]
    average_transaction_amount: float
    is_high_risk: bool


@dataclass
class Device:
    device_id: str
    ip_address: str
    is_mobile: bool


class EntityGenerator:
    """
    Generates and maintains pools of entities for transaction generation.

    Attributes:
        customers (List[Customer]): Pool of customers.
        merchants (List[Merchant]): Pool of merchants.
        devices (List[Device]): Pool of devices.
        _faker (Faker): Faker instance for generating data.
    """

    def __init__(
        self,
        num_customers: int = 100,
        num_merchants: int = 50,
        num_devices: int = 200,
        seed: Optional[int] = None,
    ):
        """
        Initialize the entity generator with configurable pool sizes.

        Args:
            num_customers: Number of unique customers to generate.
            num_merchants: Number of unique merchants to generate.
            num_devices: Number of unique devices to generate.
            seed: Random seed for reproducibility.
        """
        if seed:
            random.seed(seed)
            Faker.seed(seed)

        self._faker = Faker()
        self.customers = self._generate_customers(num_customers)
        self.merchants = self._generate_merchants(num_merchants)
        self.devices = self._generate_devices(num_devices)

        # Predefined mappings
        self.customer_to_devices = self._assign_devices_to_customers()
        self.customer_to_merchants = self._assign_merchants_to_customers()

    def _generate_customers(self, num: int) -> List[Customer]:
        """Generate a list of customers."""
        customers = []
        for _ in range(num):
            country = self._faker.country()
            city = self._faker.city()
            lat, lng = self._faker.latitude(), self._faker.longitude()
            customers.append(
                Customer(
                    customer_id=str(uuid.uuid4()),
                    name=self._faker.name(),
                    email=self._faker.email(),
                    account_age_days=random.randint(1, 3650),
                    credit_score=random.randint(300, 850),
                    typical_countries=[country],
                    typical_categories=self._generate_typical_categories(),
                    typical_transaction_types=[
                        "purchase",
                        "withdrawal",
                        "transfer",
                    ],
                    base_location=(float(lat), float(lng)),
                    average_transaction_amount=round(
                        random.uniform(20, 500), 2
                    ),
                )
            )
        return customers

    def _generate_merchants(self, num: int) -> List[Merchant]:
        """Generate a list of merchants."""
        categories = [
            "grocery",
            "electronics",
            "restaurant",
            "travel",
            "clothing",
            "entertainment",
            "healthcare",
            "fuel",
        ]
        merchants = []
        for _ in range(num):
            lat, lng = self._faker.latitude(), self._faker.longitude()
            merchants.append(
                Merchant(
                    merchant_id=str(uuid.uuid4()),
                    name=self._faker.company(),
                    category=random.choice(categories),
                    country=self._faker.country(),
                    city=self._faker.city(),
                    location=(float(lat), float(lng)),
                    average_transaction_amount=round(
                        random.uniform(10, 1000), 2
                    ),
                    is_high_risk=random.random() < 0.1,  # 10% high risk
                )
            )
        return merchants

    def _generate_devices(self, num: int) -> List[Device]:
        """Generate a list of devices."""
        devices = []
        for _ in range(num):
            devices.append(
                Device(
                    device_id=str(uuid.uuid4()),
                    ip_address=self._faker.ipv4(),
                    is_mobile=random.random() < 0.6,
                )
            )
        return devices

    def _generate_typical_categories(self) -> List[str]:
        """Generate a list of typical merchant categories for a customer."""
        all_categories = [
            "grocery",
            "electronics",
            "restaurant",
            "travel",
            "clothing",
            "entertainment",
            "healthcare",
            "fuel",
        ]
        # Each customer has 3-5 typical categories
        num_categories = random.randint(3, 5)
        return random.sample(all_categories, num_categories)

    def _assign_devices_to_customers(self) -> Dict[str, List[Device]]:
        """Randomly assign devices to customers."""
        mapping = {}
        for customer in self.customers:
            # Each customer has 1-3 devices
            num_devices = random.randint(1, 3)
            assigned = random.sample(self.devices, num_devices)
            mapping[customer.customer_id] = assigned
        return mapping

    def _assign_merchants_to_customers(self) -> Dict[str, List[Merchant]]:
        """Assign merchants that a customer frequently shops at."""
        mapping = {}
        for customer in self.customers:
            # Filter merchants by categories typical for this customer
            typical_merchants = [
                m
                for m in self.merchants
                if m.category in customer.typical_categories
            ]
            if not typical_merchants:
                typical_merchants = self.merchants
            # Assign 5-15 merchants
            num_merchants = min(
                len(typical_merchants), random.randint(5, 15)
            )
            assigned = random.sample(typical_merchants, num_merchants)
            mapping[customer.customer_id] = assigned
        return mapping

    def get_random_customer(self) -> Customer:
        """Return a random customer."""
        return random.choice(self.customers)

    def get_random_merchant(self, customer: Customer) -> Merchant:
        """Return a random merchant appropriate for the given customer."""
        merchants = self.customer_to_merchants[customer.customer_id]
        return random.choice(merchants)

    def get_random_merchant_not_typical(self, customer: Customer) -> Merchant:
        """Return a random merchant that is not typical for the customer."""
        typical = set(self.customer_to_merchants[customer.customer_id])
        non_typical = [m for m in self.merchants if m not in typical]
        if not non_typical:
            return random.choice(self.merchants)
        return random.choice(non_typical)

    def get_random_device(self, customer: Customer) -> Device:
        """Return a random device associated with the customer."""
        devices = self.customer_to_devices[customer.customer_id]
        return random.choice(devices)

    def get_random_device_not_typical(self, customer: Customer) -> Device:
        """Return a random device not typically used by the customer."""
        typical = set(self.customer_to_devices[customer.customer_id])
        non_typical = [d for d in self.devices if d not in typical]
        if not non_typical:
            return random.choice(self.devices)
        return random.choice(non_typical)

    def get_location(self, country: Optional[str] = None) -> Tuple[float, float, str, str]:
        """
        Generate a geographic location.
        If country is provided, try to generate location within that country
        (simplified: use country's capital city coordinates from Faker).
        """
        if country:
            # Simplified: use Faker's local lat/lng for the country
            lat = self._faker.latitude()
            lng = self._faker.longitude()
            city = self._faker.city()
            return float(lat), float(lng), country, city
        else:
            country = self._faker.country()
            city = self._faker.city()
            lat, lng = self._faker.latitude(), self._faker.longitude()
            return float(lat), float(lng), country, city