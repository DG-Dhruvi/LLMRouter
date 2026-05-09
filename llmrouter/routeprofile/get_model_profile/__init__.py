from llmrouter.routeprofile.get_model_profile.training_free import (
    build_flat_profile,
    build_emb_gnn_profile,
    build_index_profile,
)
from llmrouter.routeprofile.get_model_profile.trainable import (
    build_trainable_gnn_profile,
)

__all__ = [
    "build_flat_profile",
    "build_emb_gnn_profile",
    "build_index_profile",
    "build_trainable_gnn_profile",
]

try:
    from llmrouter.routeprofile.get_model_profile.training_free import build_text_gnn_profile
    __all__.append("build_text_gnn_profile")
except ImportError:
    pass
