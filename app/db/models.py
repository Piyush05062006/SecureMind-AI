from sqlalchemy import Column, Integer, String, Boolean, DateTime
from app.db.database import Base
import datetime

class Incident(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, index=True) 
    timestamp = Column(DateTime, default=datetime.datetime.utcnow) 
    video_path = Column(String(255)) 
    is_false_alarm = Column(Boolean, default=False)
    agent_reasoning = Column(String(1000))