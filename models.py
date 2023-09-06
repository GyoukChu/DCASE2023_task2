import torch
import torch.nn as nn
import torchvision

import models_mae

import models_vit
import timm
from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import PatchEmbed

class ResNext50_32x4d(nn.Module):
    def __init__(self, n_class):
        super(ResNext50_32x4d, self).__init__()

        # Accept single channel image (i.e. spectrogram) & any output number of classes
        pretrained_resnext = torchvision.models.resnext50_32x4d(weights='IMAGENET1K_V2')
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
        with torch.no_grad():
            new_conv.weight[:, 0] = pretrained_resnext.conv1.weight.mean(1)
        pretrained_resnext.conv1 = new_conv
        
        pretrained_resnext.fc = nn.Linear(in_features=2048, out_features=n_class)
        nn.init.kaiming_uniform_(pretrained_resnext.fc.weight, nonlinearity='relu')

        self.resnext = pretrained_resnext
    
    def forward(self, x):
        out = self.resnext(x)
        return out

class ConvNeXt_Base(nn.Module):
    def __init__(self, n_class):
        super(ConvNeXt_Base, self).__init__()

        # Accept single channel image (i.e. spectrogram) & any output number of classes
        pretrained_convnext = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        new_conv = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        with torch.no_grad():
            new_conv.weight[:, 0] = pretrained_convnext._modules['features'][0][0].weight.mean(1)
            pretrained_convnext._modules['features'][0][0] = new_conv
            
        pretrained_convnext._modules['classifier'][2] = nn.Linear(in_features=1024, out_features=n_class)
        nn.init.kaiming_uniform_(pretrained_convnext._modules['classifier'][2].weight)

        self.convnext = pretrained_convnext
    
    def forward(self, x):
        out = self.convnext(x)
        return out

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class AudioMAE_pretrained(nn.Module):
    def __init__(self, n_class):
        super(AudioMAE_pretrained, self).__init__()

        self.model = models_vit.__dict__['vit_base_patch16'](
            num_classes=n_class,
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=True,
            use_custom_patch=False
        )

        img_size = (1024, 128)
        in_chans = 1
        emb_dim = 768
        self.model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
        num_patches = self.model.patch_embed.num_patches
        #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        self.model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
        
        finetune = "./checkpoint/pretrained.pth"
        checkpoint = torch.load(finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % finetune)
        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        
        trunc_normal_(self.model.head.weight, std=2e-5)
        '''
        for name, param in self.model.named_parameters():
            if "head" not in name:
                param.requires_grad=False
        '''
    
    def forward(self, x):
        return self.model(x)