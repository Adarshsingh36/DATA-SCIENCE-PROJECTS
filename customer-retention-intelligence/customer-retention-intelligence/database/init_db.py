"""Create all tables (idempotent):  python -m database.init_db"""
from database.connection import get_engine, check_connection
from database.models import metadata

def init_db(drop: bool = False):
    eng = get_engine()
    if drop:
        metadata.drop_all(eng)
    metadata.create_all(eng)
    return check_connection()

if __name__ == "__main__":
    print(init_db())
