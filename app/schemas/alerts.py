from datetime import date, datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.models.alerts import AlertType

# Same rules for creating and changing a threshold: greater than 0, at most 4 decimals
Threshold = Annotated[Decimal, Field(gt=0, max_digits=18, decimal_places=4)]


class AlertCreate(BaseModel):
    ticker: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=15)]
    alert_type: AlertType
    threshold: Threshold


class AlertUpdate(BaseModel):
    threshold: Threshold | None = None
    active: bool | None = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> "AlertUpdate":
        # An empty PATCH body would change nothing, so it is rejected with 422
        if self.threshold is None and self.active is None:
            raise ValueError("Provide threshold, active or both")
        return self


class AlertResponse(BaseModel):
    # Built from an Alert model; ticker and company_name are properties of the model.
    # org_id and user_id are deliberately not included
    model_config = ConfigDict(from_attributes=True)

    id: int
    ticker: str
    company_name: str
    alert_type: str
    threshold: float
    active: bool
    watch_from: date
    created_at: datetime
