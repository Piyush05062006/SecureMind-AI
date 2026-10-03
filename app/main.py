from fastapi import FastAPI
from app.api import camera, incidents
from app.db.database import engine
from app.db import models


models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Autonomous Security API")


app.include_router(camera.router, prefix="/api/camera", tags=["Camera Control"])
app.include_router(incidents.router, prefix="/api/incidents", tags=["Incident Logs"])

@app.get("/")
def health_check():
    return {"status": "System Online", "version": "1.0"}