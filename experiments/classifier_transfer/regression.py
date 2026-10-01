"""Run the frozen scientific trainer with an explicitly recorded CNN encoder."""

from .models import CNNRegressor, CNN_NAME, DILATIONS


def install_cnn_adapter():
    from enhancer_pleiotropy_model import training
    original_metadata = training.architecture_metadata
    original_load = training.load_config

    def metadata(model, contexts):
        result = original_metadata(model, contexts)
        result.update(name=CNN_NAME, sequence_mixer="dilated convolution, no attention",
                      dilations_bins=list(DILATIONS), encoder_receptive_field_bp=2102,
                      relative_position_max_distance_bins=0)
        return result

    def load(path):
        config = original_load(path)
        if (config["model"].get("encoder_variant") != CNN_NAME or config["model"]["preset"] != "4x"
                or config["training"]["loss"]["name"] != "alphagenome_profile_cross_context"):
            raise ValueError("CNN wrapper requires its explicit configuration")
        return config

    # Process-local adapter: no modification of the frozen package or live parent.
    training.EnformerLikeJointProfileRegressor = CNNRegressor
    training.architecture_metadata = metadata
    training.load_config = load
    return training


if __name__ == "__main__": install_cnn_adapter().main()
