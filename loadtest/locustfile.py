from locust import HttpUser, task, between

class SentimentUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict_sentiment(self):
        payload = {"text": "المنتج ممتاز جدا والتوصيل سريع وممتاز"}
        self.client.post("/predict", json=payload)
