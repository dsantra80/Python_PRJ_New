from flask import Flask
from routes import main
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
app.register_blueprint(main)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
