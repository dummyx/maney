from __future__ import annotations

from typing import Literal

from nlp_trader.research_agents.contracts import (
    ExperimentTemplateSpace,
    ParameterChoice,
)

TEMPLATE_ID = "matched_feature_ablation_v1"
TEMPLATE_VERSION = "1"


def compile_template_patch(
    template: ExperimentTemplateSpace,
    choices: tuple[ParameterChoice, ...],
) -> tuple[tuple[Literal["features.text_decay_half_life_days"], int | float | str | bool], ...]:
    """Map only frozen typed parameters to known ResearchConfig fields."""

    if template.template_id != TEMPLATE_ID or template.version != TEMPLATE_VERSION:
        raise ValueError("only matched_feature_ablation_v1 version 1 is implemented")
    ranges = {value.parameter_id: value for value in template.parameters}
    provided = {value.parameter_id: value.value for value in choices}
    if set(provided) != set(ranges):
        raise ValueError("template proposal must choose every frozen parameter exactly once")
    compiled: list[
        tuple[Literal["features.text_decay_half_life_days"], int | float | str | bool]
    ] = []
    for parameter_id, parameter in sorted(ranges.items()):
        value = provided[parameter_id]
        if parameter.value_type == "integer" and type(value) is not int:
            raise ValueError("integer template parameter has the wrong type")
        if parameter.value_type == "number" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError("number template parameter has the wrong type")
        if parameter.value_type == "string" and not isinstance(value, str):
            raise ValueError("string template parameter has the wrong type")
        if parameter.value_type == "boolean" and type(value) is not bool:
            raise ValueError("boolean template parameter has the wrong type")
        if parameter.allowed_values and value not in parameter.allowed_values:
            raise ValueError("template parameter is outside its allowed values")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if parameter.minimum is not None and value < parameter.minimum:
                raise ValueError("template parameter is below its frozen minimum")
            if parameter.maximum is not None and value > parameter.maximum:
                raise ValueError("template parameter exceeds its frozen maximum")
        if parameter_id == "text_decay_days":
            compiled.append(("features.text_decay_half_life_days", value))
        else:
            raise ValueError("template contains an unmapped parameter")
    return tuple(compiled)
