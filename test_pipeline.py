"""Run the test on full pipeline with both time and space sampler
"""
import os
import json

import numpy as np
import torch
from tqdm import tqdm

import src.utils.logging as logging
from src.utils.metrics import AverageMeter, accuracy, multitask_accuracy

logger = logging.get_logger(__name__)


def test(model, device, test_loader, args, has_groundtruth):
    # Switch model to eval mode
    model.eval()
    # model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpus).to(device)

    # Test
    with torch.no_grad():
        if has_groundtruth:
            logger.info('Testing on val...')
            results = test_with_gt(model, device, test_loader)
            torch.save(results, os.path.join(args.logdir, 'results'))
        else:
            for split, loader in test_loader.items():
                logger.info('Testing on split: {}...'.format(split))
                results, extra_results = test_without_gt(model, device, loader)

                with open(os.path.join(args.logdir, split+'.json'), 'w') as outfile:
                    json.dump(results, outfile)
                torch.save(extra_results, os.path.join(args.logdir, split+'.extra'))


def test_with_gt(model, device, test_loader):
    """Test on the validation set with groundtruth labels
    """
    model_name = type(model.module).__name__
    assert model_name in ['Pipeline6', 'Pipeline8', 'Pipeline9']
    assert test_loader.dataset.name == 'epic_kitchens', \
        'Unsupported dataset: {}'.format(test_loader.dataset.dataset_name)

    # Prepare metrics
    top1 = AverageMeter()
    top5 = AverageMeter()
    verb_top1 = AverageMeter()
    verb_top5 = AverageMeter()
    noun_top1 = AverageMeter()
    noun_top5 = AverageMeter()
    all_skip, all_time, all_ssim, all_gflops = [], [], [], []
    all_output = []

    # Test
    multihead = type(model.module.actreg_model).__name__ in ['ActregGRU3']
    for i, (sample, target) in enumerate(test_loader):
        # Forward
        sample = {k: v.to(device) for k, v in sample.items()}
        target = {k: v.to(device) for k, v in target.items()}
        if model_name == 'Pipeline6':
            output, extra_output = model(sample)
        elif model_name in ['Pipeline8', 'Pipeline9']:
            output, _, _, gflops = model(sample)

        # Collect extra results
        if model_name == 'Pipeline6':
            all_skip.append(extra_output['skip'])
            all_time.append(extra_output['time'])
            if isinstance(extra_output['ssim'], np.ndarray):
                all_ssim.append(extra_output['ssim'])
            else:
                all_ssim.append(extra_output['ssim'].cpu().numpy())
        elif model_name in ['Pipeline8', 'Pipeline9']:
            all_gflops.append(gflops)
        all_output.append(output)

        # Parse output
        if not multihead:
            verb_output = output[0]
            noun_output = output[1]
        else:
            n_heads = len(output) // 2
            verb_output, noun_output = None, None
            for h in range(n_heads):
                _out, _weight = output[2*h], output[2*h+1]
                _weight = _weight[0][0]
                verb_output = _out[0] if verb_output is None else verb_output+_out[0]
                noun_output = _out[1] if noun_output is None else noun_output+_out[1]
            verb_output /= n_heads
            noun_output /= n_heads

        # Compute metrics
        batch_size = verb_output.shape[0]

        verb_prec1, verb_prec5 = accuracy(verb_output, target['verb'], topk=(1, 5))
        verb_top1.update(verb_prec1, batch_size)
        verb_top5.update(verb_prec5, batch_size)

        noun_prec1, noun_prec5 = accuracy(noun_output, target['noun'], topk=(1, 5))
        noun_top1.update(noun_prec1, batch_size)
        noun_top5.update(noun_prec5, batch_size)

        prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                          (target['verb'], target['noun']),
                                          topk=(1, 5))
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)

        # Print intermediate results
        if (i % 100 == 0) and (i != 0):
            msg = '[{}/{}]\n'.format(i, len(test_loader))
            msg += '  Prec@1 {:.3f}, Prec@5 {:.3f}\n'.format(top1.avg, top5.avg)
            msg += '  Verb Prec@1 {:.3f}, Verb Prec@5 {:.3f}\n'.format(verb_top1.avg, verb_top5.avg)
            msg += '  Noun Prec@1 {:.3f}, Noun Prec@5 {:.3f}'.format(noun_top1.avg, noun_top5.avg)
            logger.info(msg)

    # Print out message
    msg = 'Overall results:\n'
    msg += '  Prec@1 {:.3f}, Prec@5 {:.3f}\n'.format(top1.avg, top5.avg)
    msg += '  Verb Prec@1 {:.3f}, Verb Prec@5 {:.3f}\n'.format(verb_top1.avg, verb_top5.avg)
    msg += '  Noun Prec@1 {:.3f}, Noun Prec@5 {:.3f}\n'.format(noun_top1.avg, noun_top5.avg)
    if model_name == 'Pipeline6':
        all_skip = np.concatenate(all_skip, axis=0)
        all_ssim = np.concatenate(all_ssim, axis=0)
        all_time = np.concatenate(all_time, axis=0)
        msg += '  Total frames {}, Skipped frames {}'.format(all_skip.size, all_skip.sum())
    logger.info(msg)

    # Collect metrics
    test_metrics = {'top1': top1.avg,
                    'top5': top5.avg,
                    'verb_top1': verb_top1.avg,
                    'verb_top5': verb_top5.avg,
                    'noun_top1': noun_top1.avg,
                    'noun_top5': noun_top1.avg,
                    }
    results = {
        'test_metrics': test_metrics,
        'all_output': all_output,
    }

    if model_name == 'Pipeline6':
        results.update({
            'all_skip': all_skip,
            'all_ssim': all_ssim,
            'all_time': all_time,
        })
    elif model_name in ['Pipeline8', 'Pipeline9']:
        results.update({
            'all_gflops': torch.cat(all_gflops, dim=0).cpu().detach().numpy(),
            'gflops_full': model.module.gflops_full,
            'gflops_prescan': model.module.gflops_prescan,
        })

        n_skipped = (results['all_gflops'] == 0).sum()
        n_prescanned = (results['all_gflops'] == model.module.gflops_prescan).sum()
        n_nonskipped = (results['all_gflops'] == model.module.gflops_full).sum()
        msg = '\n'
        msg += '  Skipped frames     {}\n'.format(n_skipped)
        msg += '  Prescanned frames  {}\n'.format(n_prescanned)
        msg += '  Non-skipped frames {}\n'.format(n_nonskipped)
        logger.info(msg)
    return results


def test_without_gt(model, device, test_loader):
    """Test on the test set without groundtruth labels
    """
    model_name = type(model.module).__name__
    assert model_name in ['Pipeline5', 'Pipeline6', 'Pipeline8', 'Pipeline9']
    assert test_loader.dataset.name == 'epic_kitchens', \
        'Unsupported dataset: {}'.format(test_loader.dataset.dataset_name)

    # Prepare for json output
    uid_lst = test_loader.dataset.list_file.index.values
    results = {
        "version": "0.1",
        "challenge": "action_recognition",
        "results": {},
    }

    # Test
    multihead = type(model.module.actreg_model).__name__ in ['ActregGRU3']
    all_skip, all_time, all_ssim, all_gflops = [], [], [], []
    cnt = 0
    for i, (sample, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Inference
        sample = {k: v.to(device) for k, v in sample.items()}
        if model_name == 'Pipeline5':
            output = model(sample)
        elif model_name == 'Pipeline6':
            output, extra_output = model(sample)
        elif model_name in ['Pipeline8', 'Pipeline9']:
            output, _, _, gflops = model(sample)

        # Parse outputs
        if not multihead:
            verb_output = output[0]
            noun_output = output[1]
        else:
            n_heads = len(output) // 2
            verb_output, noun_output = None, None
            for h in range(n_heads):
                _out, _weight = output[2*h], output[2*h+1]
                _weight = _weight[0][0]
                verb_output = _out[0] if verb_output is None else verb_output+_out[0]
                noun_output = _out[1] if noun_output is None else noun_output+_out[1]
            verb_output /= n_heads
            noun_output /= n_heads
        verb_output = verb_output.cpu().numpy()
        noun_output = noun_output.cpu().numpy()

        # Collect prediction
        batch_size = verb_output.shape[0]
        for b in range(batch_size):
            uid = str(uid_lst[cnt])
            results["results"][uid] = {
                'verb': {str(k): float(verb_output[b][k]) for k in range(len(verb_output[b]))},
                'noun': {str(k): float(noun_output[b][k]) for k in range(len(noun_output[b]))},
            }
            cnt += 1

        # Collect extra results
        if model_name == 'Pipeline6':
            all_skip.append(extra_output['skip'])
            all_time.append(extra_output['time'])
            if isinstance(extra_output['ssim'], np.ndarray):
                all_ssim.append(extra_output['ssim'])
            else:
                all_ssim.append(extra_output['ssim'].cpu().numpy())
        elif model_name in ['Pipeline8', 'Pipeline9']:
            all_gflops.append(gflops)

    # Print out message
    extra_results = None
    if model_name == 'Pipeline6':
        msg = '  Total frames {}, Skipped frames {}'.format(len(all_skip), sum(all_skip))
        logger.info(msg)

        all_skip = np.concatenate(all_skip, axis=0)
        all_ssim = np.concatenate(all_ssim, axis=0)
        all_time = np.concatenate(all_time, axis=0)
        extra_results = {
            'all_skip': all_skip,
            'all_ssim': all_ssim,
            'all_time': all_time,
        }
    elif model_name in ['Pipeline8', 'Pipeline9']:
        all_gflops = torch.cat(all_gflops, dim=0)

        extra_results = {
            'total_gflops': all_gflops.sum().item(),
            'avg_gflops': all_gflops.mean().item(),
            'n_skipped': (all_gflops == 0).sum().item(),
            'n_prescanned': (all_gflops == model.module.gflops_prescan).sum().item(),
            'n_nonskipped': (all_gflops == model.module.gflops_full).sum().item(),
            'all_gflops': all_gflops,
        }

        msg = '\n'
        msg += '  Skipped frames     {}\n'.format(extra_results['n_skipped'])
        msg += '  Prescanned frames  {}\n'.format(extra_results['n_prescanned'])
        msg += '  Non-skipped frames {}\n'.format(extra_results['n_nonskipped'])
        logger.info(msg)

    return results, extra_results
