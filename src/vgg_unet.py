
"""
Created on Fri Jun 22 16:08:41 2018

@author: caiom
"""
import torch
import torchvision
import numpy as np

class UNetVgg(torch.nn.Module):
    """
    BorderNetwork is a NN that aims to detected border and classify occlusion.
    The architecture is a VGG without the last pool layer. After that we
    have two paths, one for regression and one for classification (occlusion).
    """

    def __init__(self, nClasses, pretrained=False):
        super(UNetVgg, self).__init__()

        vgg16pre = torchvision.models.vgg16(pretrained=pretrained)
        self.vgg0 = torch.nn.Sequential(*list(vgg16pre.features.children())[:4])
        self.vgg1 = torch.nn.Sequential(*list(vgg16pre.features.children())[4:9])
        self.vgg2 = torch.nn.Sequential(*list(vgg16pre.features.children())[9:16])
        self.vgg3 = torch.nn.Sequential(*list(vgg16pre.features.children())[16:23])
        self.vgg4 = torch.nn.Sequential(*list(vgg16pre.features.children())[23:30])



        self.smooth0 = torch.nn.Sequential(
                torch.nn.Conv2d(128 + 4, 64, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True)
                )
        self.smooth1 = torch.nn.Sequential(
                torch.nn.Conv2d(256 + 4, 64, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True)
                )
        self.smooth2 = torch.nn.Sequential(
                torch.nn.Conv2d(512 + 4, 128, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True)
                )
        self.smooth3 = torch.nn.Sequential(
                torch.nn.Conv2d(1024 + 4, 256, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True), 
                torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True)
                )
        #self.attentionmodule0 = AttentionModule(128, 128)
        #self.attentionmodule2 = AttentionModule(512, 512)
        #self.attentionmodule3 = AttentionModule(1024, 1024)
        #self.attentionmodule1 = AttentionModule(256, 256)
        
        


        self.final = torch.nn.Conv2d(64, nClasses, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x (torch.tensor): A tensor of size (batch, 3, H, W)
        Returns:
            reg_out (torch.tensor): A tensor with results of the regression (batch, 4).
            cls_out (torch.tensor): A tensor with results of the classification (batch, 2).
        """

        feat0 = self.vgg0(x)
        feat1 = self.vgg1(feat0)
        feat2 = self.vgg2(feat1)
        feat3 = self.vgg3(feat2)
        feat4 = self.vgg4(feat3)

        B,_,H,W = feat3.size()
        up3 = torch.nn.functional.interpolate(feat4, size=(H,W), mode='bilinear', align_corners=False)
        Y, X = torch.meshgrid([torch.arange(0, [H,W][i], 1.0).float()/ float([H,W][i]) + torch.rand(1) for i in range(2)])
        YX = torch.cat(B*[torch.cat([X.unsqueeze(-1),Y.unsqueeze(-1)],2).unsqueeze(0)]).float().to(up3.device)
        YX = YX.permute(0, 3, 1, 2)
        feat3 = torch.cat([feat3,YX],1)
        up3 = torch.cat([up3,YX],1)
        
        
        concat3 = torch.cat([feat3, up3], 1)
        #_, concat3 = self.attentionmodule3(concat3)
        end3 = self.smooth3(concat3)

        B,_,H,W = feat2.size()
        up2 = torch.nn.functional.interpolate(end3, size=(H,W), mode='bilinear', align_corners=False)
        Y, X = torch.meshgrid([torch.arange(0, [H,W][i], 1.0).float()/ float([H,W][i]) + torch.rand(1) for i in range(2)])
        YX = torch.cat(B*[torch.cat([X.unsqueeze(-1),Y.unsqueeze(-1)],2).unsqueeze(0)]).float().to(up2.device)
        YX = YX.permute(0, 3, 1, 2)
        feat2 = torch.cat([feat2,YX],1)
        up2 = torch.cat([up2,YX],1)

        concat2 = torch.cat([feat2, up2], 1)
        #_ , concat2 = self.attentionmodule2(concat2)
        end2 = self.smooth2(concat2)

        B,_,H,W = feat1.size()
        up1 = torch.nn.functional.interpolate(end2, size=(H,W), mode='bilinear', align_corners=False)
        Y, X = torch.meshgrid([torch.arange(0, [H,W][i], 1).float()/ float([H,W][i]) + torch.rand(1) for i in range(2)])
        YX = torch.cat(B*[torch.cat([X.unsqueeze(-1),Y.unsqueeze(-1)],2).unsqueeze(0)]).float().to(up1.device)
        YX = YX.permute(0, 3, 1, 2)
        feat1 = torch.cat([feat1,YX],1)
        up1 = torch.cat([up1,YX],1)
        
        
        concat1 = torch.cat([feat1, up1], 1)
        #_, concat1 = self.attentionmodule1(concat1)
        end1 = self.smooth1(concat1)

        B,_,H,W = feat0.size()
        up0 = torch.nn.functional.interpolate(end1, size=(H,W), mode='bilinear', align_corners=False)
        
        Y, X = torch.meshgrid([torch.arange(0, [H,W][i], 1).float()/ float([H,W][i]) + torch.rand(1) for i in range(2)])
        YX = torch.cat(B*[torch.cat([X.unsqueeze(-1),Y.unsqueeze(-1)],2).unsqueeze(0)]).float().to(up0.device)
        YX = YX.permute(0, 3, 1, 2)
        feat0 = torch.cat([feat0,YX],1)
        up0 = torch.cat([up0,YX],1)
                
        
        concat0 = torch.cat([feat0, up0], 1)
        #_, concat0 = self.attentionmodule0(concat0)
        end0 = self.smooth0(concat0)

        return self.final(end0)


    @staticmethod
    def get_params_by_kind(model, n_base = 7):

        base_vgg_bias = []
        base_vgg_weight = []
        core_weight = []
        core_bias = []

        for name, param in model.named_parameters():
            if 'vgg' in name and ('weight' in name or 'bias' in name):
                vgglayer = int(name.split('.')[-2])

                if vgglayer <= n_base:
                    if 'bias' in name:
                        print('Adding %s to base vgg bias.' % (name))
                        base_vgg_bias.append(param)
                    else:
                        base_vgg_weight.append(param)
                        print('Adding %s to base vgg weight.' % (name))
                else:
                    if 'bias' in name:
                        print('Adding %s to core bias.' % (name))
                        core_bias.append(param)
                    else:
                        print('Adding %s to core weight.' % (name))
                        core_weight.append(param)

            elif ('weight' in name or 'bias' in name):
                if 'bias' in name:
                    print('Adding %s to core bias.' % (name))
                    core_bias.append(param)
                else:
                    print('Adding %s to core weight.' % (name))
                    core_weight.append(param)

        return (base_vgg_weight, base_vgg_bias, core_weight, core_bias)

# End class

