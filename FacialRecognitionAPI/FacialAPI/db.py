# storing peoples embeddings some SQL Lite
import json
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# db.py
import json
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Person(Base):
    __tablename__ = "people"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    surname = Column(String, nullable=True)
    embeddings_json = Column(Text, nullable=True)  # JSON list of embeddings (each encoding is a list of floats)
    image_path = Column(String, nullable=True)  # optional stored image path

    def get_embeddings(self):
        if not self.embeddings_json:
            return []
        return json.loads(self.embeddings_json)

    def set_embeddings(self, embeddings: List[List[float]]):
        self.embeddings_json = json.dumps(embeddings)

# Setup
engine = create_engine("sqlite:///faces.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)