"""Process-local flatten/dense adapter for the unchanged frozen scientific trainer."""

from .models import DenseRegressor, DENSE_NAME


def install_dense_adapter():
    from enhancer_pleiotropy_model import training
    original_metadata = training.architecture_metadata
    original_load = training.load_config

    def metadata(model, contexts):
        result = original_metadata(model, contexts)
        result.update(name=DENSE_NAME, sequence_mixer="flatten all 128 positions, then two dense layers",
                      relative_position_max_distance_bins=0, encoder_receptive_field_bp=2048,
                      flattened_dimension=24576, dense_dimensions=[256, 256],
                      dense_block="Linear-LayerNorm-GELU-Dropout(0.1)",
                      flatten_order="position, channel", input_length_bp=2048,
                      heads={"atac": "Linear(256,256)-reshape(32,8)-Softplus",
                             "h3k27ac": "Linear(256,192)-reshape(24,8)-Softplus"},
                      h3k27ac_output_pooling="direct dense output at 64 bp; no latent pooling",
                      fixed_target_masks_bp={"atac": [768, 1280], "h3k27ac": [256, 1792]})
        return result

    def load(path):
        config = original_load(path)
        if (config["model"].get("encoder_variant") != DENSE_NAME or config["model"]["preset"] != "4x"
                or config["training"]["loss"]["name"] != "alphagenome_profile_cross_context"):
            raise ValueError("Dense wrapper requires its explicit configuration")
        return config

    training.EnformerLikeJointProfileRegressor = DenseRegressor
    training.architecture_metadata = metadata
    training.load_config = load
    return training


if __name__ == "__main__":
    install_dense_adapter().main()
