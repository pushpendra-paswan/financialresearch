from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ToolCallResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    step: int
    tool_name: str
    input: dict
    output: str
    is_error: bool
    approval_status: str
    duration_ms: int


class AgentRunResponse(BaseModel):
    # org_id and user_id are deliberately not included
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    step_count: int
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    message_id: int
    answer_message_id: int | None
    # Oldest first
    tool_calls: list[ToolCallResponse]
