from hnet.sft.trainer import SFTTrainConfig, build_training_arguments


def test_training_arguments_use_pytorch_checkpoints_for_tied_weights(tmp_path):
    config = SFTTrainConfig(
        model_config_path="model.json",
        pretrained_model_path="checkpoint.pt",
        output_dir=str(tmp_path),
        max_steps=1,
    )

    arguments = build_training_arguments(config)

    assert arguments.save_safetensors is False
