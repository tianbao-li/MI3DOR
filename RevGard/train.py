from data import *
from net import *
import datetime
from tqdm import tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb
cudnn.benchmark = True
cudnn.deterministic = True
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()

num_classes = [i for i in range(args.data.dataset.n_classes)]
n_views = args.data.dataset.n_views

if args.misc.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    gpu_ids = select_GPUs(args.misc.gpus)
    output_device = gpu_ids[0]


now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = f'{args.log.root_dir}/{now}'
print(log_dir)

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc,
    'alexnet': AlexnetFc
}

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(num_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        return y, d


totalNet = TotalNet()

feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids, output_device=output_device).train(True)

# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=20000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)


global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0


while global_step < args.train.min_step:

    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = im_source.to(output_device)
        im_target = im_target.to(output_device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        #---------- view pooling
        fc1_t = fc1_t.view((int(fc1_t.shape[0]/n_views), n_views, fc1_t.shape[-1]))
        fc1_t = torch.max(fc1_t, 1)[0].view(fc1_t.shape[0], -1)
        #----------

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        domain_prob_discriminator_source = discriminator.forward(feature_source)
        domain_prob_discriminator_target = discriminator.forward(feature_target)
		
        # ==============================compute loss
        adv_loss = torch.zeros(1, 1).to(output_device)
        adv_loss_separate = torch.zeros(1, 1).to(output_device)

        tmp = nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)
        tmp = nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)
		    
        # ============================== cross entropy loss, it receives logits as its inputs
        ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source)
        ce = torch.mean(ce, dim=0, keepdim=True)
				

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_discriminator]):
            loss = ce + adv_loss
            loss.backward()

        if global_step % 50 == 0:
            print("Global_step: {:05d},  Loss: {:.08f},  ce loss: {:.08f}, \
            adv_loss: {:.08f}".format(global_step, loss.cpu().detach().numpy()[0][0], \
            ce.cpu().detach().numpy()[0], adv_loss.cpu().detach().numpy()[0][0]))

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(num_classes))), variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(output_device)
            logger.add_scalar('adv_loss', adv_loss, global_step)
            logger.add_scalar('ce', ce, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)
            logger.add_scalar('loss', loss, global_step)

        if global_step % args.test.test_interval == 0:

            counters = AccuracyCounter()
            with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():

                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                    im = im.to(output_device)
                    label = label.to(output_device)

                    feature = feature_extractor.forward(im)
                    #----------view pooling
                    feature = feature.view((int(feature.shape[0]/n_views), n_views, feature.shape[-1]))
                    feature = torch.max(feature, 1)[0].view(feature.shape[0], -1)
                    label = label[::n_views]
                    #----------
                    __, __, before_softmax, predict_prob = classifier.forward(feature)


            counters = [AccuracyCounter() for x in range(len(num_classes))]

            for (each_predict_prob, each_label) in zip(predict_prob, label):                       
                if each_label in num_classes:
                    counters[each_label].Ntotal += 1.0
                    each_pred_id = np.argmax(each_predict_prob.cpu())
                    if each_pred_id == each_label.cpu():
                        counters[each_label].Ncorrect += 1.0

            counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.num_classes)))

            acc_test = counter.reportAccuracy()
	    logger.add_scalar('acc_test', acc_test, global_step)
            clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'discriminator': discriminator.state_dict() if not isinstance(discriminator, Nonsense) else 1.0,
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            print("Global_step: {:05d}, acc_test: {}, best_acc: {}".format(global_step, acc_test, best_acc))

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)

