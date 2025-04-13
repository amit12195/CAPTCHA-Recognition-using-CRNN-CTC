import string
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from recognition_utils.utils import CTCLabelConverter, AttnLabelConverter
from recognition_utils.dataset import RawDataset, AlignCollate
from recognition_utils.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------- CONFIG SECTION -------------------

class Opt:
    # Required paths
    image_folder = 'result/crops'
    #saved_model = 'TPS-ResNet-BiLSTM-Attn.pth'
    saved_model = 'ai_models/best_accuracy.pth'
    # Model architecture
    Transformation = 'TPS'
    FeatureExtraction = 'ResNet'
    SequenceModeling = 'BiLSTM'
    Prediction = 'CTC'#'Attn'

    # Input/output settings
    imgH = 32
    imgW = 100
    rgb = False
    character = '0123456789abcdefghijklmnopqrstuvwxyz'
    sensitive = False  # Set to True to use full printable characters
    PAD = False

    # Training/inference settings
    batch_max_length = 25
    num_fiducial = 20
    input_channel = 1
    output_channel = 512
    hidden_size = 256
    batch_size = 192
    workers = 4

    # Device settings
    num_gpu = torch.cuda.device_count()

opt = Opt()

# Optional: enable 94-char mode if needed
if opt.sensitive:
    opt.character = string.printable[:-6]

cudnn.benchmark = True
cudnn.deterministic = True

# ------------------- DEMO FUNCTION -------------------

def demo(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    print('Loading model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=AlignCollate_demo, pin_memory=True)

    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            log = open('./result/recognition_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
            log.close()

# ------------------- MAIN -------------------
if __name__ == '__main__':
    demo(opt)
