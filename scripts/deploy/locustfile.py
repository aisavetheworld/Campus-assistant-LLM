"""Locust load test for Campus Assistant /chat endpoint."""

import random
from locust import HttpUser, task, between

QUERIES = [
    "Can I start CPT before my advisor approves it?",
    "How do I waive UC SHIP health insurance?",
    "My package was not delivered to the mailroom. What should I do?",
    "I have an immunization hold preventing registration. How do I clear it?",
    "What happens to my GPA if I drop a class after week 2?",
    "How do I apply for OPT after graduation?",
    "Where can I find the financial aid appeal form?",
    "How do I change my major at UCSD?",
    "What is the deadline to add a class this quarter?",
    "How do I get a parking permit on campus?",
]


class ChatUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat(self):
        query = random.choice(QUERIES)
        self.client.post(
            "/chat",
            json={"query": query, "top_k": 5},
            timeout=60,
        )
