"""Register the HdLM single-path WoS format with LlamaFactory."""

from .converter import AlpacaDatasetConverter, register_dataset_converter
from .formatter import EmptyFormatter, StringFormatter
from .template import register_template


EOT = "<|eot_id|>"


def role_header(role: str) -> str:
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"


class WoSDatasetConverter(AlpacaDatasetConverter):
    def __call__(self, example):
        aligned = super().__call__(example)
        aligned["_response"][0]["content"] = (
            role_header("thought") + example["think"] + EOT
            + role_header("subtask") + example["subtask"] + EOT
            + role_header("assistant") + example[self.dataset_attr.response]
        )
        return aligned


def register_wos_data() -> None:
    register_dataset_converter("hdlm_wos", WoSDatasetConverter)
    register_template(
        name="hdlm_wos",
        format_system=StringFormatter(slots=[role_header("system") + "{{content}}" + EOT]),
        format_user=StringFormatter(slots=[role_header("user") + "{{content}}" + EOT]),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=[EOT],
        replace_eos=True,
        replace_jinja_template=True,
    )
