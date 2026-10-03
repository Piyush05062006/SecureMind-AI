from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Incident

router = APIRouter()

@router.get("/")
def get_all_incidents(db: Session = Depends(get_db)):
    # Fetch all verified alarms for the frontend
    return db.query(Incident).order_by(Incident.timestamp.desc()).all()