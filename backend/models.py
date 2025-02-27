from sqlalchemy import Column, Integer, String, Float, JSON
from database import Base

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    learning_rate = Column(Float, nullable=False)
    batch_size = Column(Integer, nullable=False)
    epochs = Column(Integer, nullable=False)
    status = Column(String, default="pending")  
    accuracy = Column(Float, nullable=True)
    train_loss = Column(Float, nullable=True) 
    val_loss = Column(Float, nullable=True) 
    epoch = Column(Integer, nullable=True)  


