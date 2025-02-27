from pydantic import BaseModel
from typing import Optional  # Import Optional

class ExperimentCreate(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int

from typing import Optional, List

class ExperimentResponse(BaseModel):
    id: int
    learning_rate: float
    batch_size: int
    epochs: int
    status: str
    accuracy: Optional[float]
    train_loss: Optional[float]  
    val_loss: Optional[float]  
    epoch: Optional[int] 

    class Config:
        from_attributes = True
