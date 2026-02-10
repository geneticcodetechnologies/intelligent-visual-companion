# to store names and faces
import sqlite3
import numpy as np

conn = sqlite3.connect("faces.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    encoding BLOB
)
""")
conn.commit()

def save_face(name, encoding):
    cursor.execute(
        "INSERT INTO faces (name, encoding) VALUES (?, ?)",
        (name, encoding.tobytes())
    )
    conn.commit()

def get_all_faces():
    cursor.execute("SELECT name, encoding FROM faces")
    rows = cursor.fetchall()

    faces = []
    for name, enc in rows:
        faces.append((name, np.frombuffer(enc, dtype=np.float64)))

    return faces
