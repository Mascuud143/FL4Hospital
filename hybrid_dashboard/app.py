import os

from flask import Flask

from hybrid.hospital_initializer import initialize_hybrid_hospital
from hybrid_dashboard.routes import hybrid_bp
from persistence import init_db


def _should_reset_database() -> bool:
    # Flask's debug reloader starts the app twice. Reset only in the parent process.
    return os.environ.get("WERKZEUG_RUN_MAIN") != "true"


def create_app() -> Flask:
    if _should_reset_database():
        db_path = os.getenv("FL4HOSPITAL_DB_PATH", "fl4hospital.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        init_db()
        initialize_hybrid_hospital()
    else:
        init_db()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "hybrid-dashboard-dev"
    app.register_blueprint(hybrid_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
