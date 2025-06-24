from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ExampleModel(BaseModel):
    """Request model for creating or updating an example."""
    title: str = Field(..., min_length=1, max_length=100, description="The title of the example")
    description: Optional[str] = Field(None, max_length=1000, description="A detailed description of the example")

    class Config:
        schema_extra = {
            "example": {
                "title": "Sample Example",
                "description": "This is a sample example description"
            }
        }


class ExampleResponse(BaseModel):
    """Response model for example data."""
    id: int = Field(..., description="The unique identifier for the example")
    title: str = Field(..., description="The title of the example")
    description: Optional[str] = Field(None, description="A detailed description of the example")
    owner: str = Field(..., description="Username of the example owner")
    created_at: datetime = Field(default_factory=datetime.now, description="When the example was created")
    updated_at: Optional[datetime] = Field(None, description="When the example was last updated")

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "title": "Sample Example",
                "description": "This is a sample example description",
                "owner": "johndoe",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-02T00:00:00Z"
            }
        }
        
        # This allows the model to be used with ORMs like SQLAlchemy
        orm_mode = True