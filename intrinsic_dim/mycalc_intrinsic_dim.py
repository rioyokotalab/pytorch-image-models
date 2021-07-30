"""
[Pope+, ICLR21] の実験を再現する
made by Yusuke Kondo
edited by Sora Takashima
"""
import argparse
from datetime import datetime
from pprint import pprint
import time
import json
import skdim
import torchvision
from torch.utils.data import Subset, DataLoader
import numpy as np
from pathlib import Path
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
# NOTE: OpenBLASが並列化でメモリを食い尽くすのを防ぐ、パフォーマンスは落ちる
# os.environ["OMP_NUM_THREADS"] = "1"
def get_args():
    parser = argparse.ArgumentParser(
        description="Estimate intrinsic dimension in many methods."
    )
    parser.add_argument("-d", "--data_dir", type=str, default="./data/")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pprint(args)
    return args
def load_datasets(data_dir: Path, is_dry_run: bool):
    # TODO: 白色化を施して比較
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    datasets = {}
    # NOTE: RuntimeError: The daily quota of the file img_align_celeba.zip is exceeded ...
    # datasets["CelebA"] = torchvision.datasets.CelebA(
    #     root=data_dir, split="test", download=True
    # )



    # datasets["MNIST"] = torchvision.datasets.MNIST(
    #     root=data_dir, train=False, transform=transform, download=True
    # )
    if not is_dry_run:
    #     datasets["CIFAR10"] = torchvision.datasets.CIFAR10(
    #         root=data_dir, train=False, transform=transform, download=True
    #     )
    #     datasets["CIFAR100"] = torchvision.datasets.CIFAR100(
    #         root=data_dir, train=False, transform=transform, download=True
    #     )
    #     datasets["SVHN"] = torchvision.datasets.SVHN(
    #         root=data_dir, split="test", transform=transform, download=True
    #     )
    #     imagenet_transform = torchvision.transforms.Compose(
    #         [
    #             torchvision.transforms.Resize(256),
    #             torchvision.transforms.CenterCrop(224),
    #             transform,
    #         ]
    #     )
    #     imagenet_dir = "/groups/gca50014/imnet/ILSVRC2012/val"
    #     datasets["ImageNet"] = torchvision.datasets.ImageFolder(
    #         root=imagenet_dir, transform=imagenet_transform
    #     )
    #     fake_dir = "/groups/gca50014/imnet/FakeImageNet1k_v1_val"
    #     datasets["Fake1kv1"] = torchvision.datasets.ImageFolder(
    #         root=fake_dir, transform=imagenet_transform
    #     )


        fractal_dir = "/groups/gcd50691/datasets/FractalDB-1k-Color"
        datasets["FractalDB1kColor"] = torchvision.datasets.ImageFolder(
            root=fractal_dir, transform=imagenet_transform
        )
        # TODO: MS-COCO の実装
    return datasets
def generate_id_estimators():
    estimators = dict()
    # ハイパーパラメータkを変更して網羅的にIDを調査
    k_cands = (3, 5, 10, 20)
    for k in k_cands:
        estimator_name = "MLE_" + str(k)
        estimators[estimator_name] = skdim.id.MLE(
            dnoise=None,
            sigma=0,
            n=None,
            integral_approximation="Haro",
            unbiased=False,
            neighborhood_based=True,
            K=k,
        )
    # estimators["KNN"] = skdim.id.KNN()
    # estimators["TwoNN"] = skdim.id.TwoNN()
    # TODO: GeoMLE、及び論文中にはなかった他のID推定手法の実装
    return estimators
def main():
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    args = get_args()
    resample_num = 10
    # NOTE: 最も画像数が少ないMNISTのvalが10000枚
    sample_num_lst = (100, 250, 500, 1000, 2000, 5000, 10000)
    data_dir = Path(args.data_dir)
    is_dry_run = args.dry_run
    datasets = load_datasets(data_dir, is_dry_run)
    estimators = generate_id_estimators()
    results = dict()
    for estimator_name, estimator in estimators.items():
        print(estimator_name)
        estimator_results = dict()
        for dataset_name, dataset in datasets.items():
            print("\t" + dataset_name)
            dataset_results = dict()
            for sample_num in sample_num_lst:
                ids = []
                for _ in range(resample_num):
                    dataset_subset = Subset(
                        dataset,
                        np.random.choice(len(dataset), sample_num, replace=False),
                    )
                    data_loader = DataLoader(dataset_subset, batch_size=sample_num)
                    all_data = next(iter(data_loader))[0]
                    all_data = all_data.view(sample_num, -1).numpy()
                    start_t = time.time()
                    estimator.fit(all_data)
                    end_t = time.time()
                    t_diff = end_t - start_t
                    _id = float(estimator.dimension_)
                    print("\t\t{}\t{}".format(_id, str(t_diff)))
                    ids += [_id]
                dataset_results[sample_num] = ids
            estimator_results[dataset_name] = dataset_results
        results[estimator_name] = estimator_results
    # Save
    pprint(results)
    output_filename = args.prefix + "_" + date_str + "_result.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    main()