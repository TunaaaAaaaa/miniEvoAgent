import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


ModelT = TypeVar("ModelT", bound="BaseModel")


class BaseModel(PydanticBaseModel):
    """Common model base for serializable miniEvoAgent components."""

    class_name: str | None = None
    version: int = 0

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        protected_namespaces=(),
    )

    def model_post_init(self, __context: Any) -> None:
        if self.class_name is None:
            object.__setattr__(self, "class_name", type(self).__name__)

    @classmethod
    def from_dict(cls: type[ModelT], data: dict[str, Any]) -> ModelT:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls: type[ModelT], content: str) -> ModelT:
        return cls.from_dict(json.loads(content))

    @classmethod
    def from_json_file(cls: type[ModelT], path: str | Path) -> ModelT:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(
        self,
        *,
        exclude_none: bool = True,
        exclude: set[str] | None = None,
        mode: str = "python",
    ) -> dict[str, Any]:
        return self.model_dump(
            exclude_none=exclude_none,
            exclude=exclude or set(),
            mode=mode,
            warnings=False,
        )

    def to_json(
        self,
        *,
        exclude_none: bool = True,
        exclude: set[str] | None = None,
        indent: int | None = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(
                exclude_none=exclude_none,
                exclude=exclude,
                mode="json",
            ),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )

    def to_config(self) -> dict[str, Any]:
        """Return a persistence-friendly component configuration."""

        return self.to_dict(mode="json")

    def save_json(
        self,
        path: str | Path,
        *,
        exclude_none: bool = True,
        exclude: set[str] | None = None,
        indent: int | None = 2,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            self.to_json(
                exclude_none=exclude_none,
                exclude=exclude,
                indent=indent,
            ),
            encoding="utf-8",
        )
        return target
