from pydantic import BaseModel, validator

class Position(BaseModel):
    S1: str | None = None
    S2: str | None = None
    S3: str | None = None
    S4: str | None = None
    M1: str | None = None
    M2: str | None = None
    
    class Config:
        validate_all = False

    @validator("M1")
    def check_m1(cls, value):
        if value is None:
            raise ValueError("M1 can't be None")
        return value

    @validator("M2")
    def check_m2(cls, value):
        if value is None:
            raise ValueError("M2 can't be None")
        return value


class Response(BaseModel):
    Resp: Position | str
    Desc: str