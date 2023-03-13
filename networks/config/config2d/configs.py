import ml_collections
import sys
sys.path.append("E:/tai_lieu_hoc_tap/tdh/tuannca_datn")
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = 'pretrain_model/transunet/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (1, 1)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = 'pretrain_model/transunet/R50_ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 9
    config.n_skip = 3
    config.activation = 'softmax'
    return config


def get_EFNB7_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (1, 1)

    config.classifier = 'seg'
    config.pretrained_path = 'runs/epoch_125_transeffunet.pth'
    config.decoder_channels = (80, 48, 32, 16, 16)
    config.skip_channels = [224, 80, 48, 32, 0]
    config.n_classes = 9
    config.n_skip = 4
    config.activation = 'softmax'
    return config