import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import cv2
from skimage import io
import numpy as np
from detection_utils import craft_utils
from detection_utils import imgproc
from detection_utils import file_utils
import json
import zipfile
from detection_utils.craft import CRAFT
from collections import OrderedDict

# Configuration - set everything here
config = {
    "trained_model": "ai_models/craft_mlt_25k.pth",
    "text_threshold": 0.7,
    "low_text": 0.4,
    "link_threshold": 0.4,
    "cuda": False,
    "canvas_size": 1280,
    "mag_ratio": 1.5,
    "poly": False,
    "show_time": False,
    "test_folder": "./input_frames/",
    "refine": False,
    "refiner_model": "weights/craft_refiner_CTW1500.pth"
}
# ---------------------- Utility Function ----------------------
# Converts DataParallel model's state_dict to single-GPU format if needed

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, config["canvas_size"], interpolation=cv2.INTER_LINEAR, mag_ratio=config["mag_ratio"]
    )
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    render_img = np.hstack((score_text.copy(), score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if config["show_time"]:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

if __name__ == '__main__':
    result_folder = './result/'
    os.makedirs(result_folder, exist_ok=True)

    image_list, _, _ = file_utils.get_files(config["test_folder"])

    net = CRAFT()
    print('Loading weights from checkpoint (' + config["trained_model"] + ')')
    if config["cuda"]:
        net.load_state_dict(copyStateDict(torch.load(config["trained_model"])))
    else:
        net.load_state_dict(copyStateDict(torch.load(config["trained_model"], map_location='cpu')))

    if config["cuda"]:
        net = net.cuda()
        net = nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    refine_net = None
    # if config["refine"]:
    #     from refinenet import RefineNet
    #     refine_net = RefineNet()
    #     print('Loading weights of refiner from checkpoint (' + config["refiner_model"] + ')')
    #     if config["cuda"]:
    #         refine_net.load_state_dict(copyStateDict(torch.load(config["refiner_model"])))
    #         refine_net = refine_net.cuda()
    #         refine_net = nn.DataParallel(refine_net)
    #     else:
    #         refine_net.load_state_dict(copyStateDict(torch.load(config["refiner_model"], map_location='cpu')))
    #     refine_net.eval()
    #     config["poly"] = True

    t = time.time()
    for k, image_path in enumerate(image_list):
        
        ##------------------------------------------------------
        image = imgproc.loadImage(image_path)
        boxes, polys, score_text = test_net(net, image, config["text_threshold"], config["link_threshold"], config["low_text"], config["cuda"], config["poly"], refine_net)

        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        # === Save cropped regions ===
        crop_folder = os.path.join(result_folder, "crops")
        os.makedirs(crop_folder, exist_ok=True)

        for i, box in enumerate(boxes):
            rect = np.array(box).astype(np.int32)
            x, y, w, h = cv2.boundingRect(rect)
            crop = image[y:y+h, x:x+w]

            crop_file = os.path.join(crop_folder, f"{filename}_box_{i+1}.jpg")
            cv2.imwrite(crop_file, crop)

        # Save result visualizations (optional)
        #file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)


    print("\nElapsed time : {:.2f}s".format(time.time() - t))
