import yaml
import easydict
from os.path import join
import scipy.io


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


import argparse
parser = argparse.ArgumentParser(description='Code for *MI3DOR*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.load(open(config_file))

save_config = yaml.load(open(config_file))

args = easydict.EasyDict(args)

dataset = None

if args.data.dataset.name == 'MI3DOR':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['source', 'target', 'source_test', 'target_test'],
    files = [
        'list_source_train.txt',
        'list_target_train.txt',
        'list_source_test.txt',
        'list_target_test.txt',
    ],
    prefix=args.data.dataset.root_path)
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')

source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
source_domain_test_name = dataset.domains[args.data.dataset.source_test]
target_domain_test_name = dataset.domains[args.data.dataset.target_test]
source_file = dataset.files[args.data.dataset.source]
target_file = dataset.files[args.data.dataset.target]
source_test_file = dataset.files[args.data.dataset.source_test]
target_test_file = dataset.files[args.data.dataset.target_test]
