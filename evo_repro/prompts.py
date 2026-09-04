from string import Formatter

from .base import BaseModel


class PromptTemplate(BaseModel):
    """Simple string template with clear missing-variable errors."""

    template: str

    def required_variables(self) -> set[str]:
        return {
            field_name
            for _, field_name, _, _ in Formatter().parse(self.template)
            if field_name
        }

    def format(self, **kwargs: object) -> str:
        missing = self.required_variables() - set(kwargs)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing prompt variable(s): {missing_list}")
        return self.template.format(**kwargs)
