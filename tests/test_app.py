import sys
import os
import pytest
from flask import Flask
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure the application root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from routes import main

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
