import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy.special import logsumexp
from skimage.filters import gaussian
from sklearn.mixture import GaussianMixture

import utils.lsun_loader as lsun_loader
from models.wrn import WideResNet
from utils.display_results import *

parser = argparse.ArgumentParser(description='Evaluates a Tiny ImageNet OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--n_components', type=int, default=50, help='Components number of GMM.')
parser.add_argument('--architecture', type=str, default='wideresnet', help='Architecture name of the DNN model.')
parser.add_argument('--dataset', type=str, default='tinyimagenet', help='In-distrubution dataset of the pretrained model.')
# Loading details
parser.add_argument('--layers', default=28, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--net_model', '-l', type=str, default='./pre_trained', help='Checkpoint path to resume / test.')
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
# Save and load
parser.add_argument('--load', type=str, help='Checkpoint path to load.')
parser.add_argument('--save', type=str, help='Checkpoint path to save.')
args = parser.parse_args()

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)

experiment_name = f'{args.architecture}_{args.dataset}'
print('Experiment:', experiment_name)

# mean and standard deviation of channels of ImageNet images
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

train_data = datasets.ImageFolder(
    root="/data/share/ood_datasets/tinyImageNet/tiny-imagenet-200/train",
    transform=test_transform)
test_data = datasets.ImageFolder(
    root="/data/share/ood_datasets/tinyImageNet/tiny-imagenet-200/val",
    transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
num_classes = 200

# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

# Restore model
assert args.net_model != ''
model_name = os.path.join(args.net_model, f'{experiment_name}.pth')
net.load_state_dict(torch.load(model_name))
net.cuda()
print('Model restored! File:', model_name)
net.eval()

if args.load:
    with open(os.path.join(args.load, f'{experiment_name}.pkl'), 'rb') as f:
        gmm_list, bigram = pickle.load(f)
else:
    # Generate features
    train_list = []
    net.eval()
    with torch.no_grad():
        # extract training data
        for itr, (input, target) in enumerate(train_loader):
            input, target = input.cuda(), target  # .cuda()
            y, out_list = net.feature_list(input)

            # process the data
            for i, layer in enumerate(out_list):
                if itr == 0:
                    print(tuple(layer.shape), '->', end=' ')
                if layer.dim() == 4:
                    layer = F.avg_pool2d(layer, layer.size(2))
                    out_list[i] = layer.reshape(layer.shape[0], -1)
                if itr == 0:
                    print(tuple(out_list[i].shape))

            # save data to list
            train_list.append([layer.cpu() for layer in out_list] + [y.cpu(), target])
            if itr % 50 == 49:
                print((itr + 1) * args.test_bs)

    train_features = [np.concatenate(f) for f in zip(*train_list)]
    n_layers = len(out_list)
    print('intermediate layers: ', n_layers)
    correct_train = np.argmax(train_features[-2], axis=1) == train_features[-1]
    print('correct number:', np.sum(correct_train))

    # train clustering
    probs_train = []
    labels_train = []
    gmm_list = []
    n_components = args.n_components
    # for layers, train...
    for i, features in enumerate(train_features):
        print('layer', i, ':', features.shape)
        if i == n_layers:  # last layer
            break

        x_train = train_features[i]

        # train kmeans
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=1000,
                                    init_params='kmeans', reg_covar=1e-6, random_state=123)
        gmm.fit(x_train)
        gmm_list.append(gmm)

        labels_train.append(gmm.predict(x_train))

    path_train = np.vstack(labels_train).T

    bigram = []
    for i in range(0, n_layers - 1):
        count = np.zeros([n_components, n_components]) + 1e-8
        for path in path_train:
            u, v = path[i], path[i + 1]
            count[u][v] += 1

        for j in range(n_components):
            count[j] /= count[j].sum()
        count = np.log(count)

        bigram.append(count)
    if args.save:
        with open(os.path.join(args.save, f'{experiment_name}.pkl'), 'wb') as f:
            print('Saving LSGM model to', args.save)
            pickle.dump([gmm_list, bigram], f)


# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data)
print('OOD examples number {}'.format(ood_num_examples))
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            # process data
            output, out_list = net.feature_list(data)
            probs_test = []
            for i, layer in enumerate(out_list):
                if layer.dim() == 4:
                    layer = F.avg_pool2d(layer, layer.size(2))
                    out_list[i] = layer.reshape(layer.shape[0], -1)
                    # get probabilities
                    probs_test.append(gmm_list[i]._estimate_weighted_log_prob(out_list[i].cpu()))

            # get scores
            scores = []
            for j in range(len(data)):
                m = probs_test[1][j].reshape(-1, 1) + probs_test[2][j].reshape(1, -1) 
                # m.shape == (k1, k2)
                m += bigram[1]

                # layer 2->3
                for i in range(3, len(gmm_list) - 1):
                    m = logsumexp(m, axis=0)
                    m = m[:, np.newaxis] + probs_test[i][j].reshape(1, -1)
                    m += bigram[i-1]  # layer i-1 -> i

                scores.append(logsumexp(m))

            smax = to_np(F.softmax(output, dim=1))
            lsgm_scores = -np.array(scores)
            _score.append(lsgm_scores)

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(lsgm_scores[right_indices])
                _wrong_score.append(lsgm_scores[wrong_indices])

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing TinyImageNet as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=experiment_name)
print_tnr95(wrong_score, right_score)

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, experiment_name)
    else:
        print_measures(auroc, aupr, fpr, experiment_name)
    print_tnr95(out_score, in_score)

# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.float32(np.clip(
    np.random.normal(size=(ood_num_examples * args.num_to_avg, 3, 64, 64), scale=0.5), -1, 1)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nGaussian Noise (sigma = 0.5) Detection')
get_and_print_results(ood_loader)

# /////////////// Rademacher Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.random.binomial(
    n=1, p=0.5, size=(ood_num_examples * args.num_to_avg, 3, 64, 64)).astype(np.float32)) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nRademacher Noise Detection')
get_and_print_results(ood_loader)

# /////////////// Blob ///////////////

ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * args.num_to_avg, 64, 64, 3)))
for i in range(ood_num_examples * args.num_to_avg):
    ood_data[i] = gaussian(ood_data[i], sigma=2, multichannel=False)
    ood_data[i][ood_data[i] < 0.75] = 0.0

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nBlob Detection')
get_and_print_results(ood_loader)

# /////////////// Textures ///////////////

ood_data = datasets.ImageFolder(root="/data/share/ood_datasets/dtd/images",
                            transform=transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64),
                                                   transforms.ToTensor(), transforms.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nTexture Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN ///////////////

ood_data = lsun_loader.LSUN("/data/share/ood_datasets/lsun/data", classes='test',
                            transform=transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64),
                                                   transforms.ToTensor(), transforms.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nLSUN Detection')
get_and_print_results(ood_loader)

# /////////////// iSUN ///////////////

ood_data = datasets.ImageFolder("/data/share/ood_datasets/iSUN",
                            transform=transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64),
                                                   transforms.ToTensor(), transforms.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\niSUN Detection')
get_and_print_results(ood_loader)

# /////////////// CIFAR Data ///////////////

ood_data = datasets.CIFAR10('/data/share/ood_datasets/cifar', train=False,
                        transform=transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64),
                                               transforms.ToTensor(), transforms.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nCIFAR-10 Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=experiment_name)
