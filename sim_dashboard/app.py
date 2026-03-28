import os

from flask import Flask

from sim_dashboard.routes import sim_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(sim_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    debug = os.getenv("FL4HOSPITAL_FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    use_reloader = os.getenv("FL4HOSPITAL_FLASK_RELOADER", "").lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug, threaded=True, use_reloader=use_reloader)
