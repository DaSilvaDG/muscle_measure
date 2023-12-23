import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import cv2
from torchsummary import summary

# from torchsummary import summary
import torch, torchvision
import glob
import random
import os.path as osp

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os, shutil, json, pathlib, copy
from albumentations import (
    HorizontalFlip,
    RandomResizedCrop,
    Compose,
    GridDistortion,
    VerticalFlip,
    Blur,
    ImageCompression,
    Resize
)


def mixup_batch(inp_batch, alpha):
    """
    Applies mixup augementation to a batch
    :param input_batch: tensor with batchsize as first dim
    :param alpha: lamda drawn from beta(alpha+1, alpha)
    """
    inp_clone = inp_batch.clone()
    # getting batch size
    batchsize = inp_batch.size()[0]

    # permute a clone
    perm = np.random.permutation(batchsize)
    for i in range(batchsize):
        inp_clone[i] = inp_batch[perm[i]]
    # generating different lambda for each sample
    # Refernced from http://www.inference.vc/mixup-data-dependent-data-augmentation/
    lam = torch.Tensor(np.random.beta(alpha + 1, alpha, batchsize))
    lam = lam.view(-1, 1, 1, 1)
    inp_mixup = lam * inp_batch + (1 - lam) * inp_clone
    return inp_mixup





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








def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        # x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


def f1_loss(y_pred, y_true, epsilon=1e-7):
    n, c, h, w = y_pred.size()
    # mask = y_true != 0

    y_true = F.one_hot(y_true, c).to(torch.float32).permute(0, 3, 1, 2)
    y_true = y_true.clamp(min=epsilon, max=1.0 - epsilon)
    # mask = mask.unsqueeze(1).expand(y_true.size()).to(torch.float32)
    y_pred = F.softmax(y_pred, dim=1)
    mask = 1
    tp = (y_true * y_pred * mask).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred) * mask).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred * mask).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred) * mask).sum().to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1.0 - epsilon)
    return 1 - (f1.mean() * precision.mean())


def dice_score(y_pred, y_true, epsilon=1e-7):
    y_true = F.one_hot(y_true, y_pred.shape[1]).to(torch.float32).permute(0, 3, 1, 2)
    y_true = y_true.clamp(min=epsilon, max=1.0 - epsilon)

    y_pred = F.softmax(y_pred, dim=1)

    intersection = (y_true * y_pred).sum(dim=0).to(torch.float32)
    dice = (2.0 * intersection) / (
        y_true.sum(dim=0).to(torch.float32) + y_pred.sum(dim=0).to(torch.float32)
    )
    return dice.mean()


def one_hot_numpy(x):
    b = np.zeros((x.size, x.max() + 1))
    b[np.arange(x.size), x] = 1
    return b

def dice_score_numpy(y_pred, y_true, axis=0, epsilon=1e-7):
    mask = (y_true != 10)
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    y_pred = one_hot_numpy(y_pred)
    y_true = one_hot_numpy(y_true)

    intersection = (y_true * y_pred).sum(axis=axis)
    dice = 2. * intersection/(
        y_true.sum(axis=axis) + y_pred.sum(axis=axis) + epsilon
    )
    return dice

def focal_loss(y_pred, y_true, alpha=1, gamma=3, epsilon=1e-7):
    y_true = F.one_hot(y_true, y_pred.shape[1]).to(torch.float32).permute(0, 3, 1, 2)
    y_true = y_true.clamp(min=epsilon, max=1.0 - epsilon)

    # y_true[:,0,:,:] = 0.0
    ce_loss = torch.nn.functional.cross_entropy(
        y_pred, y_true, reduction="none"
    )  # important to add reduction='none' to keep per-batch-item loss
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
    return focal_loss


class CSA(Dataset):
    def __init__(self, root_path="treinamento_calculo", is_train=True):
        self.json_paths = list(
            set(str(path) for path in pathlib.Path(root_path).glob("**/*json"))
        )
        if is_train:
            self.aug = Compose(
                [
                    RandomResizedCrop(
                        p=1.0,
                        height=384,
                        width=256,
                        scale=(0.8, 1.0),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    HorizontalFlip(p=0.5),
                    # VerticalFlip(p=0.5),
                    GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0),
                    Blur(p=0.5),
                    ImageCompression(
                        quality_lower=50,
                        quality_upper=80,
                    ),
                ]
            )
        else:
            self.aug = Compose(
                [
                    Resize(
                        p=1.0,
                        height=384,
                        width=256,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                ]
            )

    def get_img_name(self, path, ext=".png"):
        return path[:-5] + ext

    def get_json(self, path):
        with open(path) as file_desc:
            return json.load(file_desc)

    def get_image(self, json_path):
        for ext in [".png", ".jpg", ".tif"]:
            img_path = self.get_img_name(json_path, ext)
            img = cv2.imread(img_path)
            if img is not None:
                return img

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, key):
        json_path = self.json_paths[key]
        img = self.get_image(json_path)

        if img is None:
            print(json_path)

        mask_t = self.get_mask(json_path, img)
        augmented = self.aug(image=img, mask=mask_t)

        img_torch = torch.from_numpy(augmented["image"].transpose((2, 0, 1)))
        mask_torch = torch.from_numpy(augmented["mask"])

        return img_torch.float(), mask_torch.long()

    def get_mask(self, json_path, img):
        class2id = {"sup": 1, "femur": 2, "inf": 3, "skin": 4, "Background": 0}
        json_dict = self.get_json(json_path)

        mask = None
        for shp in json_dict["shapes"]:
            cls_id = class2id[shp["label"]]

            pontos = shp["points"]
            pontos.append(pontos[0])
            pts_np = np.array(pontos, dtype=np.int32)

            if mask is None:
                mask = np.ones((img.shape[0], img.shape[1])) *10

            cv2.drawContours(mask, [pts_np], -1, (class2id[shp["label"]],), -1)

        # cv2.drawContours(mask, [pts_np], -1, (1,), 3, lineType=cv2.LINE_AA)
        # for i in range(len(class2id)):
        #     cv2.imshow("mask", ((mask == i) * 255).astype(np.uint8))
        #     cv2.waitKey(0)

        return mask.astype(np.uint8)

    def split(self):
        train = copy.deepcopy(self)
        validation = copy.deepcopy(self)
        random.shuffle(self.json_paths)

        train.json_paths = self.json_paths[: int(0.8 * len(self))]
        validation.json_paths = self.json_paths[int(0.2 * len(self)) :]
        return train, validation


def validate(
    device,
    model,
    nb_classes,
    val_loader,
    plot_val=True,
    color_map = None
):
    if color_map is None:
        color_map = cm.get_cmap("viridis", nb_classes)
    
    for param in model.parameters():
        param.requires_grad = False

    summary(model, (3, 384, 256))

    model.eval()

    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(val_loader):
            image, gt = sample_batched
            image_np = np.squeeze(image.cpu().numpy())

            label_out = model(image.to(device))
            label_out = torch.nn.functional.softmax(label_out, dim=1)

            all_preds.append(label_out.argmax(1).cpu().numpy())
            all_gts.append(gt.cpu().numpy())
            
            gt = np.squeeze(gt.cpu().numpy())

            label_out = label_out.cpu().detach().numpy()
            label_out = np.squeeze(label_out)

            labels = np.argmax(label_out, axis=0).astype(np.uint8)
            contours, hierarchy = cv2.findContours(
                labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )[-2:]
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            image_np = (image_np / 255.0).transpose(1, 2, 0)

            for i in range(nb_classes):
                image_np[labels == i] = (
                    image_np[labels == i] + color_map.colors[i, :3] * 0.7
                )
            image_np = cv2.normalize(image_np, None, 0.0, 255.0, cv2.NORM_MINMAX)
            cv2.imshow("Image", image_np.astype("u1"))
            cv2.waitKey(300)
        all_preds = np.stack(all_preds).reshape(-1)
        all_gts = np.stack(all_gts).reshape(-1)
        from sklearn.metrics import classification_report
        # {"sup": 1, "femur": 2, "inf": 3, "skin": 4, "Background": 0}

        print(
            classification_report(
                all_gts,
                all_preds,
                labels=[0, 1, 2, 3, 4],
                target_names=["Background", "sup", "femur", "inf", "skin"],
            )
        )
        dicescore = dice_score_numpy(all_preds, all_gts)
        pretty = {k:np.around(v,4) for k, v in zip(["Background", "sup", "femur", "inf", "skin"], dicescore.tolist())}
        print(f"Dice Score per class {pretty}")
        print(f"Dice Score mean {dicescore.mean():.4}")

def main():
    nb_classes = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_epochs = 20000

    # Width x Height - MUST be divisible by 32
    # Weight of each class
    batch_size_train = 8

    path_data = "./Validation"

    dataset = CSA(path_data, is_train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    model_type = "vgg16"
    #model_type = "unet_vanilla"    
    model = UNetVgg(nClasses=nb_classes) if model_type == "vgg16" else UNet(n_class=nb_classes)
    try:
        model.load_state_dict(torch.load(f"{model_type}_model.pth"))
    except Exception:
        pass

    model = model.to(device)

    validate(device, model, nb_classes, val_loader)


if __name__ == "__main__":
    main()
