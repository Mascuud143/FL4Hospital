import os

from flask import Flask

from hybrid.hospital_initializer import initialize_hybrid_hospital
from hybrid_dashboard.routes import hybrid_bp
from persistence import init_db
from persistence.database import reset_sqlite_db, sqlite_path_from_url


def _should_reset_database() -> bool:
    # Flask's deb reloader starts the app twice. Reset only in the parent process.
    return os.environ.get("WERKZEUG_RUN_MAIN") != "true"


# create and configure the Flask app
def create_app() -> Flask:
    db_url = f"sqlite:///{os.getenv('FL4HOSPITAL_DB_PATH', 'fl4hospital.db').replace(os.sep, '/')}"
    if _should_reset_database():
        db_url, used_fallback = reset_sqlite_db(db_url, fallback_label="hybrid-dashboard")
        if used_fallback:
            fallback_path = sqlite_path_from_url(db_url)
            if fallback_path is not None:
                os.environ["FL4HOSPITAL_DB_PATH"] = str(fallback_path)
                print(f"Database file is locked; using fallback SQLite file: {fallback_path}")
        init_db(db_url=db_url)
        initialize_hybrid_hospital()
    else:
        init_db(db_url=db_url)
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "hybrid-dashboard-dev"
    app.register_blueprint(hybrid_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5001)
