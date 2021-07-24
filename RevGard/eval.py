from data import *
from net import *
import datetime
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio

cudnn.benchmark = True
cudnn.deterministic = True

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()

num_classes = args.data.dataset.n_classes

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

gpu_ids = [0]
output_device = gpu_ids[0]

n_views = args.data.dataset.n_views

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc,
    'alexnet': AlexnetFc
}

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = num_classes
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        return y, d


totalNet = TotalNet()

feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(False)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(False)
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids, output_device=output_device).train(False)

if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    discriminator.load_state_dict(data['discriminator'])

    source_feature = []
    source_label = []

    for i, (im, label) in enumerate(tqdm(source_test_dl, desc='testing ')):
        im = im.to(output_device)
        label = label.to(output_device)

        feature = feature_extractor.forward(im)
        # feature, __, before_softmax, predict_prob = classifier.forward(feature)

        source_feature.extend(feature.tolist())
        source_label.extend(label.tolist())

    target_feature = []
    target_label = []

    for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
        im = im.to(output_device)
        label = label.to(output_device)

        feature = feature_extractor.forward(im)
        feature = feature.view((int(feature.shape[0]/n_views), n_views, feature.shape[-1]))
        feature = torch.max(feature, 1)[0].view(feature.shape[0], -1)
        label = label[::n_views]

        # feature, __, before_softmax, predict_prob = classifier.forward(feature)

        target_feature.extend(feature.tolist())
        target_label.extend(label.tolist())

    sio.savemat('Results_features_labels.mat', {'source_feature':source_feature, 'source_label':source_label, \
    'target_feature':target_feature, 'target_label':target_label}) 

    print('Finished Extracting Features! Source: {}, Target: {}'.format(len(source_label), len(target_label)))
