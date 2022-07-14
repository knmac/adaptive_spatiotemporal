import sys
import os

import torch
from torch.utils.data import DataLoader
from pytablewriter import MarkdownTableWriter

sys.path.insert(0, os.path.abspath('.'))

from src.utils.load_cfg import ConfigLoader
from src.factories import ModelFactory
from src.factories import DatasetFactory
from src.utils.misc import MiscUtils
from src.utils.metrics import AverageMeter, accuracy, multitask_accuracy


dataset_cfg = 'configs/dataset_cfgs/epickitchens.yaml'
train_cfg = 'configs/train_cfgs/train_san_freeze_short.yaml'

model_cfg = 'configs/model_cfgs/pipeline5_rgbspec_san19pairfreeze_actreggru3_top1_cat.yaml'
weight = 'pretrained/complete/multihead/run_pipeline5_rgbspec_san19pairfreeze_actreggru3_top1_cat/best.model'

# model_cfg = 'configs/model_cfgs/pipeline5_rgbspec_san19pairfreeze_actreggru3_top2_cat.yaml'
# weight = 'pretrained/complete/multihead/run_pipeline5_rgbspec_san19pairfreeze_actreggru3_top2_cat/best.model'

# model_cfg = 'configs/model_cfgs/pipeline5_rgbspec_san19pairfreeze_actreggru3_top3_cat.yaml'
# weight = 'pretrained/complete/multihead/run_pipeline5_rgbspec_san19pairfreeze_actreggru3_top3_cat/best.model'


class MyMetrics():
    def __init__(self):
        self.losses = AverageMeter()
        self.verb_losses = AverageMeter()
        self.noun_losses = AverageMeter()

        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        self.verb_top1 = AverageMeter()
        self.verb_top5 = AverageMeter()

        self.noun_top1 = AverageMeter()
        self.noun_top5 = AverageMeter()

        self.criterion = torch.nn.CrossEntropyLoss()

    def update(self, target, output):
        verb_output = output[0]
        noun_output = output[1]
        batch_size = verb_output.size(0)

        loss_verb = self.criterion(verb_output, target['verb'])
        loss_noun = self.criterion(noun_output, target['noun'])
        loss = (loss_verb + loss_noun)/2

        self.losses.update(loss.item(), batch_size)
        self.verb_losses.update(loss_verb.item(), batch_size)
        self.noun_losses.update(loss_noun.item(), batch_size)

        verb_prec1, verb_prec5 = accuracy(verb_output, target['verb'], topk=(1, 5))
        self.verb_top1.update(verb_prec1, batch_size)
        self.verb_top5.update(verb_prec5, batch_size)

        noun_prec1, noun_prec5 = accuracy(noun_output, target['noun'], topk=(1, 5))
        self.noun_top1.update(noun_prec1, batch_size)
        self.noun_top5.update(noun_prec5, batch_size)

        prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                          (target['verb'], target['noun']),
                                          topk=(1, 5))
        self.top1.update(prec1, batch_size)
        self.top5.update(prec5, batch_size)

    def collect_acc(self):
        return [
            self.top1.avg, self.top5.avg,
            self.verb_top1.avg, self.verb_top5.avg,
            self.noun_top1.avg, self.noun_top5.avg
        ]


def main():
    # Load configurations
    model_name, model_params = ConfigLoader.load_model_cfg(model_cfg)
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
    train_params = ConfigLoader.load_train_cfg(train_cfg)

    dataset_params.update({
        'modality': model_params['modality'],
        'num_segments': model_params['num_segments'],
        'new_length': model_params['new_length'],
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model_factory = ModelFactory()
    model = model_factory.generate(model_name, device=device, model_factory=model_factory, **model_params)
    model.load_model(weight)
    model = model.to(device)

    # Get training augmentation and transforms
    train_augmentation = MiscUtils.get_train_augmentation(model.modality, model.crop_size)
    train_transform, val_transform = MiscUtils.get_train_val_transforms(
        modality=model.modality,
        input_mean=model.input_mean,
        input_std=model.input_std,
        scale_size=model.scale_size,
        crop_size=model.crop_size,
        train_augmentation=train_augmentation,
    )

    # Data loader
    dataset_factory = DatasetFactory()
    loader_params = {
        'batch_size': train_params['batch_size'],
        'num_workers': train_params['num_workers'],
        'pin_memory': True,
    }

    val_dataset = dataset_factory.generate(dataset_name, mode='val', transform=val_transform, **dataset_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)

    # Run validation
    global_metrics, local_metrics, both_metrics = MyMetrics(), MyMetrics(), MyMetrics()
    fuse2_metrics, fuse3_metrics = MyMetrics(), MyMetrics()

    model = torch.nn.DataParallel(model, device_ids=None).to(device)
    model.eval()
    with torch.no_grad():
        for i, (sample, target) in enumerate(val_loader):
            if i % 20 == 0:
                print(i, '/', len(val_loader))
            sample = {k: v.to(device) for k, v in sample.items()}
            target = {k: v.to(device) for k, v in target.items()}

            output = model(sample)
            output_global, _, output_local, _, output_both, _ = output

            output_fuse2 = (
                (output_global[0] + output_local[0])/2.0,
                (output_global[1] + output_local[1])/2.0,
            )
            output_fuse3 = (
                (output_global[0] + output_local[0] + output_both[0])/3.0,
                (output_global[1] + output_local[1] + output_both[1])/3.0,
            )

            global_metrics.update(target, output_global)
            local_metrics.update(target, output_local)
            both_metrics.update(target, output_both)
            fuse2_metrics.update(target, output_fuse2)
            fuse3_metrics.update(target, output_fuse3)

    writer = MarkdownTableWriter(
        table_name="Analyze multi-head",
        headers=["Head", "Top1", "Top5", "Verb Top1", "Verb Top5", "Noun Top1", "Noun Top5"],
        value_matrix=[
            ["Global"] + global_metrics.collect_acc(),
            ["Local"] + local_metrics.collect_acc(),
            ["Both"] + both_metrics.collect_acc(),
            ["Fuse2"] + fuse2_metrics.collect_acc(),
            ["Fuse3"] + fuse3_metrics.collect_acc(),
        ],
    )
    writer.write_table()


if __name__ == '__main__':
    main()
