# -- coding: utf-8 --
import torch
import torch.nn as nn
from torch.nn import functional as F
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone, MultiBackbone
# from .vit_model import VisionTransformer, PatchEmbed
from timm.models.layers import trunc_normal_, DropPath
from models.fusion_vit_3branch import VisionTransformer, PyramidVisionTransformer


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


# Squeeze-and-Excitation
class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class BaseLine(nn.Module):
    def __init__(self, img_size=224, num_classes=7, depth=8):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depth = depth

        # landmark branch
        self.face_landback = MobileFaceNet([112, 112], 136)
        face_landback_checkpoint = torch.load(
            r'./models/pretrain/mobilefacenet_model_best.pth.tar',
            map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])
        # self.conv3_lm = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        for param in self.face_landback.parameters():
            param.requires_grad = False

        # CNN feature branch
        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)
        self.ir_layer1 = nn.Linear(1024, 512)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)

        # LbpHog feature branch
        # self.ir_lbphog_back = Backbone(50, 0.0, 'ir')
        # ir_lbphog_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        # self.ir_lbphog_back = load_pretrained_weights(self.ir_lbphog_back, ir_lbphog_checkpoint)
        # self.ir_layer2 = nn.Linear(1024, 512)
        # self.conv3_LH = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.face_landback2 = MobileFaceNet([112, 112], 136)
        face_landback_checkpoint = torch.load(
            r'./models/pretrain/mobilefacenet_model_best.pth.tar',
            map_location=lambda storage, loc: storage)
        self.face_landback2.load_state_dict(face_landback_checkpoint['state_dict'])
#         self.conv3_lh = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        for param in self.face_landback2.parameters():
            param.requires_grad = False

        # ViT
        self.ViT = VisionTransformer(in_chans=49, q_chanel=49, embed_dim=512,
                                     depth=depth, num_heads=8, mlp_ratio=2.,
                                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)

    def forward(self, x, x_lbp_hog):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        x_lbp_hog = F.interpolate(x_lbp_hog, size=112)
        #######
        # landmark shape: [BatchSize,512,7,7]
        x_landmark, x_landmark2, x_landmark3 = self.face_landback(x_face)
        # x_landmark = self.conv3_lm(x_landmark)
        x_landmark = x_landmark.view(B_, -1, 49).transpose(1, 2)

        #######
        # cnn feature branch
        x_cnn_feature = self.ir_back(x)
        x_cnn_feature = self.ir_layer1(x_cnn_feature)
        # x_cnn_feature = self.conv3(x_cnn_feature)
        # x_cnn_feature = x_cnn_feature.view(B_, -1, 49).transpose(1, 2)
        #######
        # lbp feature branch
        # x_lbp_feature = self.ir_lbphog_back(x_lbp_hog)
        # # x_lbp_feature = self.ir_layer2(x_lbp_feature)
        # x_lbp_feature = self.conv3_LH(x_lbp_feature)
        # x_lbp_feature = x_lbp_feature.view(B_, -1, 49).transpose(1, 2)
        x_lbp_feature, x_lbp2, xlbp3 = self.face_landback2(x_lbp_hog)
        # x_lbp_feature = self.conv3_lh(x_lbp_feature)
        x_lbp_feature = x_lbp_feature.view(B_, -1, 49).transpose(1, 2)

        y_hat = self.ViT(x_cnn_feature, x_landmark, x_lbp_feature)
        y_hat = self.se_block(y_hat)
        out = self.head(y_hat)

        return out
