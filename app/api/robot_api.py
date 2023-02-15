from typing import Optional
import requests
from utils.constants import API_BASE_URL

from model.response import Position, Response


class RobotApi():
    def __init__(self, endpoint: str = API_BASE_URL):
        self.endpoint = endpoint

    async def make_request(self, method: str, data: Optional[Position] = None) -> Response:
        try:
            if method == "POST" and not isinstance(data, Position):
                raise ValueError("Data must be an instance of Position when making a POST request")
            if method == "GET":
                with requests.get(url=self.endpoint) as response:
                    json_response = response.json()
            elif method == "POST":
                with requests.post(self.endpoint, json=[data.dict()]) as response:
                    json_response = response.json()
            else:
                raise ValueError("Invalid method")

            if response.status_code != 200:
                raise ValueError("Invalid status code")

            return Response.parse_obj(json_response[0])
        except:
            return Response(Resp="KO", Desc="")