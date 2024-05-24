import sys
import os
import pytest
from flask import Flask

# Ensure the application root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the environment variable for testing
os.environ["HUGGINGFACE_TOKEN"] = "your_huggingface_token_here"

from routes import main  # Now this should work

@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(main)
    app.config['TESTING'] = True
    client = app.test_client()
    
    yield client

def test_generate(client):
    response = client.post('/generate', json={'prompt': 'Hello, world!'})
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'generated_text' in json_data
    assert json_data['generated_text'] is not None
