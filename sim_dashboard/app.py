from flask import Flask
from sim_dashboard.routes import sim_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(sim_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
