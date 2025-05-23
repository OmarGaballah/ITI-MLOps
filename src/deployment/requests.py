from typing import Optional

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: Optional[float]  # Age can be missing
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: Optional[str]  # Cabin can be missing
    Embarked: str
    Title: str
    FamilySize: int
    IsAlone: int
    Deck: Optional[str]  # Deck can be missing
    TicketGroupSize: int
    FarePerPerson: float
    AgeBin: Optional[int]  # AgeBin can be missing
    FareBin: Optional[int]  # FareBin can be missing
    Pclass_AgeBin: Optional[int]  # Optional, might need calculation
    Sex_Pclass: str
    CabinMissing: int
    AgeMissing: int
