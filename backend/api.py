from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import Experiment
from schemas import ExperimentCreate, ExperimentResponse
from train import train_experiment
from fastapi.middleware.cors import CORSMiddleware
import threading

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/start_experiment/", response_model=ExperimentResponse)
def start_experiment(exp: ExperimentCreate, db: Session = Depends(get_db)):
    # Store experiment in the database
    new_exp = Experiment(
        learning_rate=exp.learning_rate,
        batch_size=exp.batch_size,
        epochs=exp.epochs,
        status="running"
    )
    db.add(new_exp)
    db.commit()
    db.refresh(new_exp)

    thread = threading.Thread(target=train_experiment, args=(db, new_exp.id, exp.learning_rate, exp.batch_size, exp.epochs))
    thread.start()

    return new_exp

@app.get("/experiments/", response_model=list[ExperimentResponse])
def list_experiments(db: Session = Depends(get_db)):
    return db.query(Experiment).all()
