# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import torch

import albumentations
import pretrainedmodels

import numpy as np
import torch.nn as nn

from flask import Flask
from flask import request
from flask import render_template
from torch.nn import functional as F

from flask.json import jsonify 

from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine


app = Flask(__name__)
DEVICE = "cpu"
MODEL = None


class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = torch.sigmoid(self.l0(x))
        loss = 0
        return out, loss


def predict(image_path, model):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    test_images = [image_path]
    test_targets = [0]

    test_dataset = ClassificationLoader(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    predictions = Engine.predict(
        test_loader,
        model,
        DEVICE
    )
    return np.vstack((predictions)).ravel()


@app.route("/", methods=["POST"])
def upload_predict():
    if request.method == "POST":
       
        try: 
            request_data = request.json
            img_path = request_data['path']
            pred = predict(img_path, MODEL)[0]
            value = str(pred)
            data = {'prediction': value}
            return jsonify(data)

        except : 
            return jsonify({'error':'error during prediction'})
            
    return jsonify({'result':1})


if __name__ == "__main__":
    MODEL = SEResNext50_32x4d(pretrained=None)
    MODEL.load_state_dict(torch.load("https://www.kaggleusercontent.com/kf/35193166/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..DD9AVBK19GTqccKC8PRgqg.a7_6epDl8MYzxdFro0I057WbRdQjxpQkbMeAt94cu7oy8VRsMJVhdx6Wa_uNMNJpMEsopdo_dBLd7Oc0qh3Vsizv_hMHhdmVyMAASw4JKlcifJqbdMLGsOOXivoiIlpi_vzQue1B-P8aNLQK8JxYH3mdT48pCPjtf7YWXU1u3oOWq0J1bRDIthI4oyDGxCYhvUJiEJ2WBUkoY0KsiyYl220UkgLvXqCeqGeptUhPdX-3xH1JrFjFpmjrfKCqEn4cJyBzy-hakVENx-BcXeoDkM1tQBcowZd5mVH3mM47je3grn-kV1XV7vQDApcnxRFDhPSVTxzjV3FYZas2VHO6799vGknGtRwnYUTkC4Zu8rwmE4Lum2SYto70ly96GcPGhrU_M5svI_t3ciADN4Qt4uqahv2bz_P5XYzzfPOIY53hgQJwX4dXOVS1SG6LcEg70j58vszBJcmUR4QCBoIb14yxdhvozjZgg52x6pyXkQfruSHN9c6qGQsyc4GhwuI8nDpjzsxWly9xVH07cICMy6z0cXMKNXwFXCoNHoNM3bkEPLNHrL-8iEMqznqEmwhWWHxbN3yTSQTYL2xApHNGgAARUzHGJPDejAi_lV12GLmRFNUD0N0T2T6QuSDYoCj-jkGROYWxAK6CXPbXwj3GWQ.ez47-nxB369WDPuRNn2tJA/model_fold_4.bin", map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)
    app.run(host="0.0.0.0", port=12000, debug=True)
