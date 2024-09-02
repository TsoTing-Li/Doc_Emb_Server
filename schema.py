import re

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator


class PostEmbedDoc(BaseModel):
    data_folder: str
    recreate: bool

    @model_validator(mode="after")
    def check(self: "PostEmbedDoc") -> "PostEmbedDoc":
        if bool(re.search(r"[^a-zA-Z0-9_\-\s/]+", self.data_folder)) is True:
            raise RequestValidationError(
                {
                    "messages": f"data_folder: {self.data_folder} contain invalid characters."
                }
            )
        return self
