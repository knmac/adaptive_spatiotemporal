import os
import time

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import numpy as np

from src.utils.misc import MiscUtils
import src.utils.logging as logging
from src.utils.metrics import AverageMeter, accuracy, multitask_accuracy

logger = logging.get_logger(__name__)

__DEBUG_NOBELIEF__ = False


def train_val(model, device, criterion, train_loader, val_loader, train_params, args):
    """Training and validation routine. It will call val() automatically
    """
    # =========================================================================
    # Prepare for train val
    # =========================================================================
    # Freeze stream weights (leaves only fusion and classification trainable)
    if train_params['freeze']:
        model.freeze_fn('modalities')

    # Freeze batch normalisation layers except the first
    if train_params['partialbn']:
        model.freeze_fn('partialbn_parameters')

    # Get param_groups for optimizer
    param_groups = model.get_param_groups()

    # Create optimizer
    if train_params['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, **train_params['optim_params'])
    elif train_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups, **train_params['optim_params'])
    else:
        raise NotImplementedError

    # Create scheduler
    scheduler = MultiStepLR(optimizer, train_params['lr_steps'], gamma=0.1)

    # Summary writer
    sum_writer = SummaryWriter(log_dir=args.logdir)

    # Setup training starting point
    start_epoch, _, model, best_val = _setup_training(
        model, optimizer, device, train_params, args)

    # Train with multiple GPUs
    model = torch.nn.DataParallel(model, device_ids=args.gpus).to(device)

    # =========================================================================
    # Go through all epochs
    # =========================================================================
    for epoch in range(start_epoch, train_params['n_epochs']):
        scheduler.step()

        # Training phase
        logger.info('Training...')
        run_iter = epoch * len(train_loader)
        _train_one_epoch(
            model, device, criterion, train_loader, optimizer, sum_writer,
            epoch, run_iter, train_params)

        # Validation phase
        if ((epoch + 1) % train_params['eval_freq'] == 0) or \
                ((epoch + 1) == train_params['n_epochs']):
            if len(val_loader) != 0:
                # Run validation if val_loader is valid
                logger.info('Testing...')
                val_metrics = validate(model, device, criterion, val_loader,
                                       sum_writer, run_iter+len(train_loader))

                # Remember best value of the metrics and save checkpoint
                current_val = val_metrics[args.best_metrics]
                is_best = False
                if args.best_fn == 'max':
                    if current_val > best_val:
                        best_val = current_val
                        is_best = True
                elif args.best_fn == 'min':
                    if current_val < best_val:
                        best_val = current_val
                        is_best = True
                else:
                    NotImplementedError
            else:
                # Skip validation and keep saving the model
                # Don't update best_val --> always 0
                logger.info('Skip testing and save latest model as best...')
                is_best = True

            MiscUtils.save_progress(model, optimizer, args.savedir, best_val,
                                    epoch, is_best)

        # HACK: Decay teacher forcing if possible
        if hasattr(model.module, 'decay_teacher_forcing_ratio'):
            sum_writer.add_scalar('data/tf_ratio', model.module.tf_ratio, run_iter)
            model.module.decay_teacher_forcing_ratio(epoch)
        # HACK: Decay temperature if possible
        if hasattr(model.module, 'decay_temperature'):
            sum_writer.add_scalar('data/temperature', model.module.temperature, run_iter)
            model.module.decay_temperature(epoch)

    # =========================================================================
    # Done training
    # =========================================================================
    sum_writer.close()


def _setup_training(model, optimizer, device, train_params, args):
    """Set up training model, optimizer and get starting epoch and lr

    Args:
        model: model for resuming or loading pretrained
        optimizer: optimizer for resuming
        train_params: training parameters
        args: extra runtime parameters

    Return:
        start_epoch: epoch to start the training at
        lr: learning rate to start at
        model: loaded model by `from_pretrained` or `resume`
        best_val: current best value of the selected metrics
    """
    train_mode = args.train_mode
    savedir = args.savedir

    # By default, start_epoch = 0, lr from train parameters
    start_epoch = 0
    best_val = 0
    lr = train_params['optim_params']['lr']

    # Setup training starting point
    if train_mode == 'from_scratch':
        logger.info('Start training from scratch')
    elif train_mode == 'from_pretrained':
        # Only support pretrained flow weight for now
        if 'flow' in [x.lower() for x in model.modality]:
            logger.info('Initialize Flow stream from Kinetics')
            pretrained = 'pretrained/kinetics_tsn_flow.pth.tar'
            if not os.path.isfile(pretrained):
                root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
                pretrained = os.path.join(root, pretrained)
            state_dict = torch.load(pretrained)
            for k, v in state_dict.items():
                state_dict[k] = torch.squeeze(v, dim=0)
            try:
                base_model = getattr(model, 'flow')
            except AttributeError:
                base_model = getattr(model.light_model, 'flow')
            base_model.load_state_dict(state_dict, strict=False)
        else:
            logger.info('Flow modality not being used. Start training from scratch')
    elif train_mode == 'resume':
        logger.info('Resume training from a checkpoint')
        prefix = MiscUtils.get_lastest_checkpoint(savedir)
        lr, start_epoch, best_val = MiscUtils.load_progress(model, optimizer,
                                                            device, prefix)
    else:
        raise ValueError('Unsupported train_mode: {}'.format(train_mode))
    return start_epoch, lr, model, best_val


def _compute_loss_acc(output, target, criterion, has_multihead):
    """Compute accuracy loss for single head or multi head output
    """
    if not has_multihead:
        loss_verb = criterion(output[0], target['verb'])
        loss_noun = criterion(output[1], target['noun'])
        loss = loss_verb + loss_noun  # accuracy loss
        verb_output = output[0]
        noun_output = output[1]
    else:
        n_heads = len(output) // 2
        loss_verb, loss_noun = 0, 0
        verb_output, noun_output = None, None
        for h in range(n_heads):
            _out, _weight = output[2*h], output[2*h+1]
            _weight = _weight[0][0]
            loss_verb += _weight * criterion(_out[0], target['verb'])
            loss_noun += _weight * criterion(_out[1], target['noun'])
            verb_output = _out[0] if verb_output is None else verb_output+_out[0]
            noun_output = _out[1] if noun_output is None else noun_output+_out[1]
        loss = loss_verb + loss_noun  # accuracy loss
        verb_output /= n_heads
        noun_output /= n_heads
    return loss, loss_verb, loss_noun, verb_output, noun_output


def _compute_loss_acc_nonepic(output, target, criterion, has_multihead):
    """Compute accuracy loss for single head or multi head output (non EPIC-KITCHENS)
    """
    if not has_multihead:
        loss = criterion(output, target)
        merge_output = output  # return the original output
    else:
        n_heads = len(output) // 2
        loss = 0
        merge_output = None
        for h in range(n_heads):
            _out, _weight = output[2*h], output[2*h+1]
            _weight = _weight[0][0]
            loss += _weight * criterion(_out, target)
            merge_output = _out if merge_output is None else merge_output+_out
        merge_output /= n_heads
    return loss, merge_output


def _train_one_epoch(model, device, criterion, train_loader, optimizer,
                     sum_writer, epoch, run_iter, train_params):
    """Train one single epoch
    """
    dataset = train_loader.dataset.name
    clip_gradient = float(train_params['clip_gradient'])
    has_belief = hasattr(model.module, 'compare_belief')
    has_eff = hasattr(model.module, 'compute_efficiency_loss')
    if hasattr(model.module, 'actreg_model'):
        multihead = type(model.module.actreg_model).__name__ in ['ActregGRU3']
    else:
        multihead = False
    n_extra_losses = sum([has_belief, has_eff])

    # Switch to train mode
    model.train()

    if train_params['partialbn']:
        model.module.freeze_fn('partialbn_statistics')
    if train_params['freeze']:
        model.module.freeze_fn('bn_statistics')

    # Prepare metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if dataset == 'epic_kitchens':
        verb_losses = AverageMeter()
        noun_losses = AverageMeter()
        verb_top1 = AverageMeter()
        verb_top5 = AverageMeter()
        noun_top1 = AverageMeter()
        noun_top5 = AverageMeter()
    if has_belief:
        belief_losses = AverageMeter()
    if has_eff:
        eff_losses = AverageMeter()
        usage_losses = AverageMeter()

    # =========================================================================
    # Training loop
    # =========================================================================
    end = time.time()
    for i, (sample, target) in enumerate(train_loader):
        # Skip broken batches
        if (not isinstance(target, dict)) and (len(target) == 1) and (target == -1):
            continue

        # Measure data loading time
        data_time.update(time.time() - end)

        # Place input sample on the correct device for all modalities
        for k in sample.keys():
            sample[k] = sample[k].to(device)

        # Forward -------------------------------------------------------------
        if n_extra_losses > 0:
            if has_belief:
                output, loss_belief = model(sample)
                # Average across multiple devices, if necessary
                if __DEBUG_NOBELIEF__:
                    loss_belief = torch.Tensor([-1.0])
                    loss_belief.requires_grad = False
                else:
                    loss_belief = loss_belief.mean()
            elif has_eff:
                output, loss_eff, loss_usage, _ = model(sample)
                loss_eff = loss_eff.mean()
                loss_usage = loss_usage.mean()
        else:
            output = model(sample)

        # Compute metrics -----------------------------------------------------
        batch_size = sample[model.module.modality[0]].size(0)
        if dataset != 'epic_kitchens':
            target = target.to(device)

            # loss = criterion(output, target)  # accuracy loss
            loss, output = _compute_loss_acc_nonepic(output, target, criterion, multihead)
            if has_belief and (not __DEBUG_NOBELIEF__):
                loss += loss_belief
                belief_losses.update(loss_belief.item(), batch_size)
            if has_eff:
                loss += loss_eff + loss_usage
                eff_losses.update(loss_eff.item(), batch_size)
                usage_losses.update(loss_usage.item(), batch_size)
            loss = loss / (n_extra_losses + 1)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
        else:
            target = {k: v.to(device) for k, v in target.items()}

            # Compute loss for batch
            loss, loss_verb, loss_noun, verb_output, noun_output = _compute_loss_acc(
                output, target, criterion, multihead)
            if has_belief and (not __DEBUG_NOBELIEF__):
                loss += loss_belief
                belief_losses.update(loss_belief.item(), batch_size)
            if has_eff:
                loss += loss_eff + loss_usage
                eff_losses.update(loss_eff.item(), batch_size)
                usage_losses.update(loss_usage.item(), batch_size)
            loss = loss / (n_extra_losses + 2)

            # Update batch metrics
            verb_losses.update(loss_verb.item(), batch_size)
            noun_losses.update(loss_noun.item(), batch_size)

            verb_prec1, verb_prec5 = accuracy(verb_output, target['verb'], topk=(1, 5))
            verb_top1.update(verb_prec1, batch_size)
            verb_top5.update(verb_prec5, batch_size)

            noun_prec1, noun_prec5 = accuracy(noun_output, target['noun'], topk=(1, 5))
            noun_top1.update(noun_prec1, batch_size)
            noun_top5.update(noun_prec5, batch_size)

            prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                              (target['verb'], target['noun']),
                                              topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)

        # Compute gradient and do optimizer step ------------------------------
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), clip_gradient)
            if total_norm > clip_gradient:
                logger.info('Clipping gradient: %f with coef %f' %
                            (total_norm, clip_gradient / total_norm))
        optimizer.step()
        run_iter += 1

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print out message ---------------------------------------------------
        if i % train_params['print_freq'] == 0:
            _lr = optimizer.param_groups[-1]['lr']

            sum_writer.add_scalar('data/epochs', epoch, run_iter)
            sum_writer.add_scalar('data/learning_rate', _lr, run_iter)

            msg_prefix = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            msg_prefix += 'lr={:.5f}, batch_time={:.3f}, data_time={:.3f}\n'.format(
                _lr, batch_time.avg, data_time.avg)

            log_content = {'losses': losses, 'top1': top1, 'top5': top5}
            if dataset == 'epic_kitchens':
                log_content.update({
                    'verb_losses': verb_losses, 'verb_top1': verb_top1, 'verb_top5': verb_top5,
                    'noun_losses': noun_losses, 'noun_top1': noun_top1, 'noun_top5': noun_top5,
                })
            if has_belief:
                log_content.update({'belief_losses': belief_losses})
            if has_eff:
                log_content.update({'eff_losses': eff_losses,
                                    'usage_losses': usage_losses})

            _log_message('training', sum_writer, run_iter, log_content, msg_prefix)

    # Record the gradient flow of the last batch
    grad_flow = MiscUtils.plot_grad_flow(model.module.named_parameters(), show=False)
    sum_writer.add_image('grad_flow', grad_flow, run_iter)

    # =========================================================================
    # Collect training metrics
    # =========================================================================
    if dataset != 'epic_kitchens':
        training_metrics = {'train_loss': losses.avg, 'train_acc': top1.avg}
    else:
        training_metrics = {'train_loss': losses.avg,
                            'train_noun_loss': noun_losses.avg,
                            'train_verb_loss': verb_losses.avg,
                            'train_acc': top1.avg,
                            'train_verb_acc': verb_top1.avg,
                            'train_noun_acc': noun_top1.avg}
    if has_belief:
        training_metrics.update({'train_belief_loss': belief_losses.avg})
    if has_eff:
        training_metrics.update({'train_eff_loss': eff_losses.avg,
                                 'train_usage_loss': usage_losses.avg})
    return training_metrics


def validate(model, device, criterion, val_loader, sum_writer=None, run_iter=None):
    """Validate a trained model
    """
    dataset = val_loader.dataset.name
    has_belief = hasattr(model.module, 'compare_belief')
    has_eff = hasattr(model.module, 'compute_efficiency_loss')
    if hasattr(model.module, 'actreg_model'):
        multihead = type(model.module.actreg_model).__name__ == 'ActregGRU3'
    else:
        multihead = False
    n_extra_losses = sum([has_belief, has_eff])

    # Swith to eval mode
    model.eval()

    with torch.no_grad():
        # Prepare metrics
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if dataset == 'epic_kitchens':
            verb_losses = AverageMeter()
            noun_losses = AverageMeter()
            verb_top1 = AverageMeter()
            verb_top5 = AverageMeter()
            noun_top1 = AverageMeter()
            noun_top5 = AverageMeter()
        if has_belief:
            belief_losses = AverageMeter()
        if has_eff:
            eff_losses = AverageMeter()
            usage_losses = AverageMeter()
            all_gflops = []

        # =====================================================================
        # Validation loop
        # =====================================================================
        end = time.time()
        for i, (sample, target) in enumerate(val_loader):
            # Skip broken batches
            if (not isinstance(target, dict)) and (len(target) == 1) and (target == -1):
                continue

            # Place input sample on the correct device for all modalities
            for k in sample.keys():
                sample[k] = sample[k].to(device)

            # Forward ---------------------------------------------------------
            if n_extra_losses > 0:
                if has_belief:
                    output, loss_belief = model(sample)
                    if __DEBUG_NOBELIEF__:
                        loss_belief = torch.Tensor([-1.0])
                        loss_belief.requires_grad = False
                    else:
                        loss_belief = loss_belief.mean()
                elif has_eff:
                    output, loss_eff, loss_usage, gflops = model(sample)
                    all_gflops.append(gflops)
                    loss_eff = loss_eff.mean()
                    loss_usage = loss_usage.mean()
            else:
                output = model(sample)

            # Compute metrics -------------------------------------------------
            batch_size = sample[model.module.modality[0]].size(0)
            if dataset != 'epic_kitchens':
                # if output.ndim == 3:
                #     output = output[:, -1, :]
                target = target.to(device)
                # loss = criterion(output, target)  # accuracy loss
                loss, output = _compute_loss_acc_nonepic(output, target, criterion, multihead)
                if has_belief and (not __DEBUG_NOBELIEF__):
                    loss += loss_belief
                    belief_losses.update(loss_belief.item(), batch_size)
                if has_eff:
                    loss += loss_eff + loss_usage
                    eff_losses.update(loss_eff.item(), batch_size)
                    usage_losses.update(loss_usage.item(), batch_size)
                loss = loss / (n_extra_losses + 1)
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
            else:
                # Pick the last frame to validate
                # if output[0].ndim == 3:
                #     output = (output[0][:, -1, :], output[1][:, -1, :])
                target = {k: v.to(device) for k, v in target.items()}

                # Compute loss for batch
                loss, loss_verb, loss_noun, verb_output, noun_output = _compute_loss_acc(
                    output, target, criterion, multihead)
                if has_belief and (not __DEBUG_NOBELIEF__):
                    loss += loss_belief
                    belief_losses.update(loss_belief.item(), batch_size)
                if has_eff:
                    loss += loss_eff + loss_usage
                    eff_losses.update(loss_eff.item(), batch_size)
                    usage_losses.update(loss_usage.item(), batch_size)
                loss = loss / (n_extra_losses + 2)

                # Update batch metrics
                verb_losses.update(loss_verb.item(), batch_size)
                noun_losses.update(loss_noun.item(), batch_size)

                verb_prec1, verb_prec5 = accuracy(verb_output, target['verb'], topk=(1, 5))
                verb_top1.update(verb_prec1, batch_size)
                verb_top5.update(verb_prec5, batch_size)

                noun_prec1, noun_prec5 = accuracy(noun_output, target['noun'], topk=(1, 5))
                noun_top1.update(noun_prec1, batch_size)
                noun_top5.update(noun_prec5, batch_size)

                prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                                  (target['verb'], target['noun']),
                                                  topk=(1, 5))

            losses.update(loss.item(), batch_size)
            top1.update(prec1, batch_size)
            top5.update(prec5, batch_size)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if has_eff:
            all_gflops = torch.cat(all_gflops, dim=0)

        # Viz hallucination for Pipeline3 -------------------------------------
        if (type(model.module).__name__ == 'Pipeline3') and (torch.cuda.device_count() == 1):
            logger.info('Visualizing hallucination...')
            n_samples = min(val_loader.batch_size, 10)  # cap the number of samples to viz at 10
            indices = np.arange(0, len(val_loader.dataset),
                                int(np.ceil(len(val_loader.dataset) / n_samples))).tolist()
            _sample, _ = MiscUtils.get_samples_from_loader(val_loader, indices)
            _sample = {k: v.to(device) for k, v in _sample.items()}
            model(_sample)  # Forward to compute hallucination

            for idx in range(len(indices)):
                _rgbt = MiscUtils.deprocess_rgb(_sample['RGB'][idx], model.module.num_segments)
                _attn = model.module._attn[idx].cpu().detach().numpy().mean(axis=1)
                _hallu = model.module._hallu[idx].cpu().detach().numpy().mean(axis=1)
                _img = MiscUtils.viz_sequence([_rgbt, _attn, _hallu], viz=False)
                sum_writer.add_image(f'Hallucination_{indices[idx]}',
                                     MiscUtils.fig2img(_img), run_iter)

        # =====================================================================
        # Print out message
        # =====================================================================
        msg_prefix = 'Testing results:\n'
        log_content = {'losses': losses, 'top1': top1, 'top5': top5}

        if dataset == 'epic_kitchens':
            log_content.update({
                'verb_losses': verb_losses, 'verb_top1': verb_top1, 'verb_top5': verb_top5,
                'noun_losses': noun_losses, 'noun_top1': noun_top1, 'noun_top5': noun_top5,
            })
        if has_belief:
            log_content.update({'belief_losses': belief_losses})
        if has_eff:
            log_content.update({
                'eff_losses': eff_losses,
                'usage_losses': usage_losses,
                'total_gflops': all_gflops.sum().item(),
                'avg_gflops': all_gflops.mean().item(),
                'n_skipped': (all_gflops == 0).sum().item(),
                'n_prescanned': (all_gflops == model.module.gflops_prescan).sum().item(),
                'n_nonskipped': (all_gflops == model.module.gflops_full).sum().item(),
            })
        _log_message('validation', sum_writer, run_iter, log_content, msg_prefix)

        # =====================================================================
        # Collect validation metrics
        # =====================================================================
        if dataset != 'epic_kitchens':
            val_metrics = {'val_loss': losses.avg, 'val_acc': top1.avg}
        else:
            val_metrics = {'val_loss': losses.avg,
                           'val_noun_loss': noun_losses.avg,
                           'val_verb_loss': verb_losses.avg,
                           'val_acc': top1.avg,
                           'val_verb_acc': verb_top1.avg,
                           'val_noun_acc': noun_top1.avg}
        if has_belief:
            val_metrics.update({'val_belief_loss': belief_losses.avg})
        if has_eff:
            val_metrics.update({
                'val_eff_loss': eff_losses.avg,
                'val_usage_loss': usage_losses.avg,
                'total_gflops': all_gflops.sum().item(),
                'avg_gflops': all_gflops.mean().item(),
                'n_skipped': (all_gflops == 0).sum().item(),
                'n_prescanned': (all_gflops == model.module.gflops_prescan).sum().item(),
                'n_nonskipped': (all_gflops == model.module.gflops_full).sum().item(),
            })
        return val_metrics


def _log_message(phase, sum_writer, run_iter, data, msg_prefix=''):
    """Wrapper to print message and writer summary"""
    msg = msg_prefix

    if (sum_writer is not None) and (run_iter is not None):
        sum_writer.add_scalars('data/loss', {phase: data['losses'].avg}, run_iter)
        sum_writer.add_scalars('data/prec/top1', {phase: data['top1'].avg}, run_iter)
        sum_writer.add_scalars('data/prec/top5', {phase: data['top5'].avg}, run_iter)

    msg += '  Loss {:.4f}, Prec@1 {:.3f}, Prec@5 {:.3f}\n'.format(
        data['losses'].avg, data['top1'].avg, data['top5'].avg)

    try:
        if (sum_writer is not None) and (run_iter is not None):
            if 'verb_top1' in data.keys():
                sum_writer.add_scalars('data/verb/loss', {phase: data['verb_losses'].avg}, run_iter)
                sum_writer.add_scalars('data/noun/loss', {phase: data['noun_losses'].avg}, run_iter)
                sum_writer.add_scalars('data/verb/prec/top1', {phase: data['verb_top1'].avg}, run_iter)
                sum_writer.add_scalars('data/verb/prec/top5', {phase: data['verb_top5'].avg}, run_iter)
                sum_writer.add_scalars('data/noun/prec/top1', {phase: data['noun_top1'].avg}, run_iter)
                sum_writer.add_scalars('data/noun/prec/top5', {phase: data['noun_top5'].avg}, run_iter)

                msg += '  Verb Loss {:.4f}, Verb Prec@1 {:.3f}, Verb Prec@5 {:.3f}\n'.format(
                    data['verb_losses'].avg, data['verb_top1'].avg, data['verb_top5'].avg)
                msg += '  Noun Loss {:.4f}, Noun Prec@1 {:.3f}, Noun Prec@5 {:.3f}'.format(
                    data['noun_losses'].avg, data['noun_top1'].avg, data['noun_top5'].avg)
            if 'belief_losses' in data.keys():
                sum_writer.add_scalars('data/belief/loss', {phase: data['belief_losses'].avg}, run_iter)
            if 'eff_losses' in data.keys():
                sum_writer.add_scalars('data/eff/loss', {phase: data['eff_losses'].avg}, run_iter)
                sum_writer.add_scalars('data/usage/loss', {phase: data['usage_losses'].avg}, run_iter)
            if 'total_gflops' in data.keys():
                sum_writer.add_scalars('data/gflops/total', {phase: data['total_gflops']}, run_iter)
                sum_writer.add_scalars('data/gflops/avg', {phase: data['avg_gflops']}, run_iter)
                sum_writer.add_scalars('data/n_frames/skipped', {phase: data['n_skipped']}, run_iter)
                sum_writer.add_scalars('data/n_frames/prescanned', {phase: data['n_prescanned']}, run_iter)
                sum_writer.add_scalars('data/n_frames/nonskipped', {phase: data['n_nonskipped']}, run_iter)
    except KeyError:
        pass

    if 'belief_losses' in data.keys():
        msg += '\n  Belief Loss {:.4f}'.format(data['belief_losses'].avg)
    if 'eff_losses' in data.keys():
        msg += '\n  Eff Loss {:.4f}'.format(data['eff_losses'].avg)
        msg += '\n  Usage Loss {:.4f}'.format(data['usage_losses'].avg)
    if 'total_gflops' in data.keys():
        msg += '\n  GFLOPS: accumulated {:.4f}  avg_perframe {:.4f}'.format(
            data['total_gflops'], data['avg_gflops'])
        msg += '\n  N frames: skipped {}  prescanned {}  nonskipped {}'.format(
            data['n_skipped'], data['n_prescanned'], data['n_nonskipped'])

    logger.info(msg)
