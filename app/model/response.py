from pydantic import BaseModel, validator

class Position(BaseModel):
    S1: str | None = None
    S2: str | None = None
    S3: str | None = None
    S4: str | None = None
    M1: str | None = None
    M2: str | None = None
    

    def to_array(self):
        array = []
        if self.S1 != None:
            array.append(int(self.S1))
        else:
            return []
        if self.S2 != None:
            array.append(int(self.S2))
        else:
            return []
        if self.S3 != None:
            array.append(int(self.S3))
        else:
            return []
        if self.S4 != None:
            array.append(int(self.S4))
        else:
            return []
        if self.M1 != None:
            array.append(int(self.M1))
        else:
            array.append(0)
        if self.M2 != None:
            array.append(int(self.M2))
        else:
            array.append(0)
        return array if len(array) == 6 else []

    def from_array(self, array):
        if len(array) != 6:
            raise ValueError("The array doesn't match the required number of values")
        self.S1 = str(array[0])
        self.S2 = str(array[1])
        self.S3 = str(array[2])
        self.S4 = str(array[3])
        self.M1 = str(array[4])
        self.M2 = str(array[5])


class Response(BaseModel):
    Resp: Position | str
    Desc: str