from locust import HttpUser, task, between
import random

class MLAPIUser(HttpUser):
    wait_time = between(1, 3)  # Random wait between requests
    
    def on_start(self):
        """Initialize test data when user starts"""
        self.test_prompts = [
            "What are the symptoms of",
            "How to treat",
            "What causes",
            "Define the term",
            "Explain the concept of"
        ]
        # Test health endpoint on startup
        self.client.get("/")
    
    @task(3)
    def test_basic_inference(self):
        """Most common case: basic inference request"""
        prompt = random.choice(self.test_prompts)
        with self.client.post(
            "/infer",
            params={"prompt": prompt},
            catch_response=True
        ) as response:
            if not response.text:
                response.failure("Empty response received")
    
    @task(2)
    def test_custom_length_inference(self):
        """Test inference with custom max length"""
        prompt = random.choice(self.test_prompts)
        with self.client.post(
            "/infer",
            params={
                "prompt": prompt,
                "max_length": random.randint(50, 200)
            },
            catch_response=True
        ) as response:
            if not response.text:
                response.failure("Empty response received")
    
    @task(1)
    def test_rapid_inference(self):
        """Stress test with minimal wait time"""
        prompt = random.choice(self.test_prompts)
        self.client.post("/infer", params={"prompt": prompt})
    
    @task(1)
    def test_metrics(self):
        """Check metrics endpoint"""
        self.client.get("/metrics") 