from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import Incident

router = APIRouter()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
@router.get("/")
def get_all_incidents(db: Session = Depends(get_db)):
    return db.query(Incident).filter(Incident.is_false_alarm == False).order_by(Incident.timestamp.desc()).all()