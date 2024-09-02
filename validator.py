from pathlib import Path

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator


class PostEmbedDoc(BaseModel):
    data_folder: str

    @model_validator(mode="after")
    def check(self: "PostEmbedDoc") -> "PostEmbedDoc":
        if not (Path("upload_pdf") / self.data_folder).exists():
            raise RequestValidationError(
                {"messages": f"data_folder: {self.data_folder} does not exists."}
            )
        return self
