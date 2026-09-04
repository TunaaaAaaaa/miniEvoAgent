from string import Formatter

from .base import BaseModel


class PromptTemplate(BaseModel):
    """Simple string template with clear missing-variable errors."""

    template: str

    def required_variables(self) -> set[str]:
        return self._required_variables(self.template)

    def _required_variables(self, template: str) -> set[str]:
        variables: set[str] = set()
        for _, field_name, format_spec, _ in Formatter().parse(template):
            if field_name is None:
                continue
            if not field_name:
                raise ValueError("PromptTemplate supports named variables only.")
            root_name = field_name.split(".", 1)[0].split("[", 1)[0]
            if root_name.isdecimal():
                raise ValueError("PromptTemplate supports named variables only.")
            variables.add(root_name)
            if format_spec:
                variables.update(self._required_variables(format_spec))
        return variables

    def format(self, **kwargs: object) -> str:
        missing = self.required_variables() - set(kwargs)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing prompt variable(s): {missing_list}")
        return self.template.format(**kwargs)
