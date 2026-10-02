from datetime import date

from pydantic import BaseModel


class FilingResponse(BaseModel):
    id: int
    accession_number: str
    form_type: str
    filed_on: date
    report_date: date | None
    fiscal_year: int | None
    primary_document: str
    # raw_path is internal and is not exposed. This only says whether the file is on disk
    document_downloaded: bool


class FilingListResponse(BaseModel):
    items: list[FilingResponse]
    total: int
    page: int
    page_size: int
