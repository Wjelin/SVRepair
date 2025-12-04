from dataclasses import dataclass


@dataclass
class ModelConfig:
    use_struct_attn_bias: bool = True
    use_auxiliary_task: bool = True
    max_distance: int = 3
    struct_bias_init: str = "default"

    PRESETS = {
        "full": {},
        "only_struct_attn_bias": {
            "use_struct_attn_bias": True,
            "use_auxiliary_task": False
        },
        "only_aux_task": {
            "use_struct_attn_bias": False,
            "use_auxiliary_task": True
        },
        "baseline": {
            "use_struct_attn_bias": False,
            "use_auxiliary_task": False
        }
    }

    @classmethod
    def from_preset(cls, name: str) -> 'ModelConfig':
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {name}. Available: {list(cls.PRESETS.keys())}")
        return cls(**cls.PRESETS[name])
