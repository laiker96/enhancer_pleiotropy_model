"""Process-local enhancerNet adapter; frozen loss/training code stays unchanged."""

import json
from pathlib import Path
import sys

from .enhancernet import MODELS, NAMES, UPSTREAM_COMMIT


def install_enhancernet_adapter(architecture):
    from enhancer_pleiotropy_model import training
    cls = MODELS[architecture]
    original_metadata = training.architecture_metadata
    original_load = training.load_config

    def metadata(model, contexts):
        result = original_metadata(model, contexts)
        result.update(name=NAMES[architecture], upstream_repository="https://github.com/laiker96/enhancerNet",
            upstream_commit=UPSTREAM_COMMIT, input_length_bp=2048, latent_positions=125,
            convolution_block="valid Conv1d-BatchNorm-ReLU-Dropout1d(0.1)-MaxPool1d(2)",
            convolution_receptive_field_bp=58, encoder_receptive_field_bp=2042,
            unused_right_flank_bp=6,
            pooling="max pooling size 2 after every convolution; global max then mean concatenation",
            positional_encoding="absolute sinusoidal" if model.transformer_layers else "none",
            transformer_normalization="post-norm" if model.transformer_layers else "none",
            transformer_dropout=.2 if model.transformer_layers else 0.,
            relative_position_max_distance_bins=0, dense_dimensions=[256, 256],
            dense_block="Linear-Dropout(0.4)-ReLU", pooled_dimension=240,
            sequence_mixer="two transformer encoder layers" if model.transformer_layers else "none",
            heads={"atac": "Linear(256,256)-reshape(32,8)-Softplus",
                   "h3k27ac": "Linear(256,192)-reshape(24,8)-Softplus"},
            h3k27ac_output_pooling="direct dense output at 64 bp; no latent pooling",
            fixed_target_masks_bp={"atac": [768, 1280], "h3k27ac": [256, 1792]})
        return result

    def load(path):
        config = original_load(path)
        if (config["model"].get("encoder_variant") != NAMES[architecture]
                or config["model"]["preset"] != "4x"
                or config["training"]["loss"]["name"] != "alphagenome_profile_cross_context"):
            raise ValueError("enhancerNet wrapper requires its explicit configuration")
        return config

    training.EnformerLikeJointProfileRegressor = cls
    training.architecture_metadata = metadata
    training.load_config = load
    return training


if __name__ == "__main__":
    config = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
    architecture = next(k for k, v in NAMES.items() if v == config["model"]["encoder_variant"])
    install_enhancernet_adapter(architecture).main()
