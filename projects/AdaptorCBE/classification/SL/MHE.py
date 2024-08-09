# -*- coding: utf-8 -*-
import GLOB as glob
import numpy as np
import datetime
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import datasources
import datasets
import models as model_class
from models.utils.initStrategy import InitializeStrategy as InitS
from models.utils.ema import ModelEMA
from comm.base.comm import CommUtils as comm
from comm.misc import ProjectUtils as proj
from comm.schedule import ScheduleUtils as schedule
from comm.classification.criteria import AvgCounter, ClassAccuracy
from comm.classification.business import BusinessUtils as bus


def main(mark, params=None):
    args = init_args(params)
    args.num_labeled = args.train_num
    args = proj.project_setting(args, mark)
    logger = glob.get_value('logger')
    logger.print('L1', '=> experiment start, {}'.format(args.experiment))
    logger.print('L1', '=> experiment setting: {}'.format(dict(args._get_kwargs())))

    # region 1. Initialize
    # region 1.1 Data loading
    if args.dataset == 'CIFAR10C':
        datasource = datasources.__dict__['CIFAR10']()
    else:
        datasource = datasources.__dict__[args.dataset]()
    labeled_idx, unlabeled_idx = datasource.get_data(args)
    args.num_classes = datasource.num_classes
    args.name_classes = datasource.name_classes

    if args.dataset == 'CIFAR10C':
        labeled_dataset, _, test_dataset = datasets.__dict__[args.dataset](
            labeled_idx, unlabeled_idx, datasource.root, datasource.mean, datasource.std, args.noisy_std
        )
    else:
        labeled_dataset, _, test_dataset = datasets.__dict__[args.dataset](
            labeled_idx, unlabeled_idx, datasource.root, datasource.mean, datasource.std
        )
    # endregion

    # region 1.2 Dataloader initialize
    train_sampler = RandomSampler
    labeled_loader = DataLoader(labeled_dataset,
                                sampler=train_sampler(labeled_dataset),
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True)
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    # endregion

    # region 1.3 Model initialize
    model = model_class.__dict__['ClassModel'](args)
    model_ema = ModelEMA(model, args.ema_decay, args) if args.use_ema else None

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.wd},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=args.power, nesterov=args.nesterov)
    scheduler = schedule.get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)
    # endregion

    # region 1.4 Hyperparameters initialize
    args.start_epoch = 0
    args.best_acc = -1.
    args.best_acc2 = -1.
    args.best_epoch = 0
    args.best_epoch2 = 0
    args.eval_step = int(args.train_num / args.batch_size)
    args.epochs = args.total_steps
    # endregion
    # endregion

    # region 2. Iteration
    logger.print('L1', '=> task: {}@{}, epochs: {}, eval-step: {}, batch-size: {}'.format(
        args.dataset, args.num_labeled, args.epochs, args.eval_step, args.batch_size))
    for epo in range(args.start_epoch, args.epochs):
        epoTM = datetime.datetime.now()
        args.epo = epo

        # region 2.1 model training and validating
        startTM = datetime.datetime.now()
        total_loss, labeled_losses, sim_sum, wrong_sum, sim_rate = train(
            labeled_loader, model, model_ema, optimizer, scheduler, args)
        logger.print('L2', 'model training finished...', start=startTM)

        startTM = datetime.datetime.now()
        test_model = model_ema.ema if args.use_ema else model
        predsArray, test_losses, t1s, t5s, test_losses2, t1s2, t5s2 = validate(test_loader, test_model, args)
        logger.print('L2', 'model validating finished...', start=startTM)
        # endregion

        # region 2.2 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        is_best = t1s[-1] > args.best_acc
        if is_best:
            args.best_epoch = epo
            args.best_acc = t1s[-1]

        is_best2 = t1s2[-1] > args.best_acc2
        if is_best2:
            args.best_epoch2 = epo
            args.best_acc2 = t1s2[-1]
        # model storage
        checkpoint = {'current_epoch': args.epo,
                      'best_acc': args.best_acc,
                      'best_epoch': args.best_epoch,
                      'model': args.arch,
                      'model_state': model.state_dict(),
                      'optimizer_state': optimizer.state_dict(),
                      'scheduler_state': scheduler.state_dict()}

        if args.use_ema: checkpoint['model_ema_state'] = (
            model_ema.ema.module if hasattr(model_ema.ema, 'module') else model_ema.ema).state_dict()
        comm.ckpt_save(checkpoint, is_best, ckptPath='{}/ckpts'.format(args.basePath))
        logger.print('L2', 'model storage finished...', start=startTM)

        # Log data storage
        log_data = {'total_loss': total_loss,
                    'labeled_losses': labeled_losses,
                    'pl_similarity': sim_rate,
                    'pl_sim_num': sim_sum,
                    'pl_wrong_num': wrong_sum,
                    'test_losses': test_losses,
                    't1s': t1s,
                    't5s': t5s,
                    'test_losses2': test_losses2,
                    't1s2': t1s2,
                    't5s2': t5s2}
        comm.json_save(log_data, '{}/logs/logData/logData_{}.json'.format(args.basePath, epo+1), isCover=True)

        # region 2.3 output result
        # Training performance
        fmtc = '[{}/{} | pl_sim: {} ({}/ {})] total_loss: {}, x_loss: {}'
        logc = fmtc.format(format(epo + 1, '3d'),
                           format(args.epochs, '3d'),
                           format(sim_rate, '.3f'),
                           format(sim_sum, '5d'),
                           format(wrong_sum, '5d'),
                           format(total_loss, '.3f'),
                           format(labeled_losses[-1], '.5f'))
        logger.print('L1', logc)

        for stIdx in range(args.stream_num):
            fmtc = '[{}/{} | ms{}] x_loss: {}'
            logc = fmtc.format(format(epo + 1, '3d'),
                               format(args.epochs, '3d'),
                               stIdx+1,
                               format(labeled_losses[stIdx], '.5f'))
            logger.print('L2', logc)

        # Validating performance
        fmtc = '[mean ens][{}/{}] best acc: {} (epo: {}) | test_loss: {} | top1: {}, top5: {}'
        logc = fmtc.format(format(epo + 1, '3d'),
                           format(args.epochs, '3d'),
                           format(args.best_acc, '.2f'),
                           format(args.best_epoch + 1, '3d'),
                           format(test_losses[-1], '.3f'),
                           format(t1s[-1], '.2f'),
                           format(t5s[-1], '.2f'))
        logger.print('L1', logc)

        # Validating performance
        fmtc = '[mean GT ][{}/{}] best acc: {} (epo: {}) | test_loss: {} | top1: {}, top5: {}'
        logc = fmtc.format(format(epo + 1, '3d'),
                           format(args.epochs, '3d'),
                           format(args.best_acc2, '.2f'),
                           format(args.best_epoch2 + 1, '3d'),
                           format(test_losses2[-1], '.3f'),
                           format(t1s2[-1], '.2f'),
                           format(t5s2[-1], '.2f'))
        logger.print('L1', logc)

        for stIdx in range(args.stream_num):
            fmtc = '[{}/{} | ms{}] test_loss: {} | top1: {}, top5: {}'
            logc = fmtc.format(format(epo + 1, '3d'),
                               format(args.epochs, '3d'),
                               stIdx+1,
                               format(test_losses[stIdx], '.3f'),
                               format(t1s[stIdx], '.2f'),
                               format(t5s[stIdx], '.2f'))
            logger.print('L2', logc)

        # Epoch line
        time_interval = logger._interval_format(seconds=(datetime.datetime.now() - epoTM).seconds*(args.epochs - (epo+1)))
        fmtc = '[{}/{} | {}] ---------- ---------- ---------- ---------- ---------- ---------- ----------'
        logc = fmtc.format(format(epo + 1, '3d'), format(args.epochs, '3d'), time_interval)
        logger.print('L1', logc, start=epoTM)
        # endregion
    # endregion


def train(labeled_loader, model, model_ema, optimizer, scheduler, args):
    # region 1. Preparation
    labeled_iter = iter(labeled_loader)
    total_loss_counter = AvgCounter()
    labeled_loss_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    sim_num_counter = AvgCounter()
    wrong_num_counter = AvgCounter()
    # endregion

    # region 2. Training
    model.train()
    for batch_idx in range(args.eval_step):
        log_content = 'epoch: {}-{}: '.format(format(args.epo + 1, '4d'), format(batch_idx + 1, '4d'))
        optimizer.zero_grad()
        # region 2.1 Data organizing
        try:
            inputs_x, targets_x = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            inputs_x, targets_x = next(labeled_iter)

        batch_size = inputs_x.shape[0]
        inputs = inputs_x.to(args.device)
        targets_x = targets_x.to(args.device)
        # endregion

        # region 2.2 forward
        ms_preds = model(inputs)
        ms_logits_x = []
        for stIdx in range(args.stream_num):
            ms_logits_x.append(ms_preds[stIdx])
        del ms_preds
        ms_logits_x = torch.stack(ms_logits_x, dim=0)
        # endregion

        # region 2.3 supervised learning loss
        labeled_loss_sum = torch.tensor(0.).to(args.device)
        for stIdx in range(args.stream_num):
            labeled_loss_val = F.cross_entropy(ms_logits_x[stIdx], targets_x.long().to(args.device), reduction='none').mean()
            labeled_loss_counters[stIdx].update(labeled_loss_val.item())
            labeled_loss_sum += labeled_loss_val
        labeled_loss = labeled_loss_sum / args.stream_num
        labeled_loss_counters[-1].update(labeled_loss.item(), args.stream_num)
        log_content += ' loss_x: {}'.format(format(labeled_loss.item(), '.5f'))
        # endregion

        # region 2.4 calculate similarity of predictions of multiple heads.
        sim_num, wrong_num = bus.prediction_similarity(ms_logits_x, targets_x)
        sim_num_counter.update(sim_num)
        wrong_num_counter.update(wrong_num)
        log_content += ' | sim: {}, wrong: {}'.format(format(sim_num, '2d'), format(wrong_num, '2d'))
        # endregion

        # region 2.6 calculate total loss & update model
        loss = labeled_loss
        total_loss_counter.update(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        if args.use_ema: model_ema.update(model)
        # endregion
        # if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == args.eval_step: print(log_content)

    # region 3. calculate records
    total_loss_val = total_loss_counter.avg
    labeled_losses = [counter.avg for counter in labeled_loss_counters]
    sim_sum_val = sim_num_counter.sum
    wrong_sum_val = wrong_num_counter.sum
    sim_rate_val = sim_sum_val/wrong_sum_val if wrong_sum_val > 0 else 1.0
    # endregion
    return total_loss_val, labeled_losses, sim_sum_val, wrong_sum_val, sim_rate_val


def validate(test_loader, model, args):
    # region 1. Preparation
    test_loss_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    top1_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    top5_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    predsArray = [[] for _ in range(args.stream_num+1)]


    test_loss_counters2 = [AvgCounter() for _ in range(args.stream_num + 1)]
    top1_counters2 = [AvgCounter() for _ in range(args.stream_num + 1)]
    top5_counters2 = [AvgCounter() for _ in range(args.stream_num + 1)]
    predsArray2 = [[] for _ in range(args.stream_num+1)]

    labelsArray = []
    model.eval()
    # endregion

    # region 2. test
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # region 2.1 data organize
            inputs = inputs.to(args.device)
            targets = targets.long().to(args.device)
            labelsArray += targets.clone().cpu().data.numpy().tolist()
            # endregion

            # region 2.2 forward
            ms_logits = model(inputs)
            # endregion

            # region 2.3 calculate loss & accuracy
            prec_list = []
            for stIdx in range(args.stream_num):
                # cal test loss
                logits = ms_logits[stIdx]
                test_loss = F.cross_entropy(logits, targets)
                test_loss_counters[stIdx].update(test_loss.item(), inputs.shape[0])

                # single stream prediction accuracy
                prec1, prec5 = ClassAccuracy.accuracy(logits, targets, topk=(1, 5))
                top1_counters[stIdx].update(prec1.item(), inputs.shape[0])
                top5_counters[stIdx].update(prec5.item(), inputs.shape[0])
                prec_list.append(prec1.item())

                # single stream prediction
                _, preds = torch.max(logits, -1)
                predsArray[stIdx] += preds.clone().cpu().data.numpy().tolist()
            prec_list_tensor = torch.tensor(prec_list)
            # endregion

            # region 2.4 ensemble prediction

            # region 2.4.1 weighted mean by the rate between head prediction and ensemble prediction
            ms_logits_ens = torch.mean(ms_logits, dim=0)
            _, ms_logits_ens_preds = torch.max(ms_logits_ens, dim=-1)

            frequency = []
            for stIdx in range(args.stream_num):
                logits = ms_logits[stIdx]
                _, logits_preds = torch.max(logits, dim=-1)
                frequency.append(torch.eq(ms_logits_ens_preds, logits_preds).float().sum())

            weighted = torch.softmax(torch.stack(frequency, dim=0), dim=0)
            weighted_ms_logits = []
            for stIdx in range(args.stream_num):
                weighted_ms_logits.append(ms_logits[stIdx]*weighted[stIdx])
            weighted_ms_logits = torch.stack(weighted_ms_logits, dim=0)
            logits_ens = torch.sum(weighted_ms_logits, dim=0)
            ens_loss = F.cross_entropy(logits_ens, targets)
            test_loss_counters[-1].update(ens_loss.item(), inputs.shape[0])

            # get prediction
            _, preds_ens = torch.max(logits_ens, -1)
            predsArray[-1] += preds_ens.clone().cpu().data.numpy().tolist()

            # cal the accuracy
            prec1_ens, prec5_ens = ClassAccuracy.accuracy(logits_ens, targets, topk=(1, 5))
            top1_counters[-1].update(prec1_ens.item(), inputs.shape[0])
            top5_counters[-1].update(prec5_ens.item(), inputs.shape[0])
            # endregion

            # region 2.4.2 weighted mean by accuracy of head prediction
            # ensembling
            weighted2 = torch.softmax(prec_list_tensor, dim=0)
            weighted_ms_logits2 = []
            for stIdx in range(args.stream_num):
                weighted_ms_logits2.append(ms_logits[stIdx]*weighted2[stIdx])
            weighted_ms_logits2 = torch.stack(weighted_ms_logits2, dim=0)
            logits_ens2 = torch.sum(weighted_ms_logits2, dim=0)
            ens_loss2 = F.cross_entropy(logits_ens2, targets)
            test_loss_counters2[-1].update(ens_loss2.item(), inputs.shape[0])

            # get prediction
            _, preds_ens2 = torch.max(logits_ens2, -1)
            predsArray2[-1] += preds_ens2.clone().cpu().data.numpy().tolist()

            # cal the accuracy
            prec1_ens2, prec5_ens2 = ClassAccuracy.accuracy(logits_ens2, targets, topk=(1, 5))
            top1_counters2[-1].update(prec1_ens2.item(), inputs.shape[0])
            top5_counters2[-1].update(prec5_ens2.item(), inputs.shape[0])
            # endregion

        # region 3. calculate records
        test_losses = [counter.avg for counter in test_loss_counters]
        top1s = [counter.avg for counter in top1_counters]
        top5s = [counter.avg for counter in top5_counters]

        test_losses2 = [counter.avg for counter in test_loss_counters2]
        top1s2 = [counter.avg for counter in top1_counters2]
        top5s2 = [counter.avg for counter in top5_counters2]
        # endregion

        # region 4 prediction distribution visualization
        if args.debug and (args.epo+1) % 5 == 0 and (batch_idx + 1) == args.eval_step:
            dist_box = bus.target_statistic_infer(args, torch.from_numpy(np.array(predsArray[-1])), torch.from_numpy(np.array(labelsArray)))
            save_path = '{}/{}/distribution_visualization/e{}_b{}.jpg'.format(glob.expr, args.experiment, args.epo+1, batch_idx+1)
            proj.distribution_visualize(dist_box, save_path)
        # endregion
    return predsArray, test_losses, top1s, top5s, test_losses2, top1s2, top5s2


def init_args(params=None):
    parser = argparse.ArgumentParser(description='FixMatch Training')

    # Model Setting
    parser.add_argument('--arch', default='WideResNet_MHE', type=str, help='model name')
    parser.add_argument('--use-ema', default='True')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')

    # Mulitple Stream
    parser.add_argument('--stream-num', default=5, type=int, help='number of stream')
    parser.add_argument('--noisy-factor', default=0.0, type=float)

    # Dataset setting
    parser.add_argument('--dataset', default='CIFAR10C', choices=['CIFAR10', 'CIFAR100', 'ANIMAL10N', 'CIFAR10C'], help='dataset name')
    parser.add_argument('--noisy-std', default=2.0, type=float)
    parser.add_argument('--train-num', default=50000, type=int, help='number of total training data')
    parser.add_argument('--valid-num', default=10000, type=int, help='number of validating data')

    # Training strategy
    parser.add_argument('--total-steps', default=200, type=int, help='number of total steps to run')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')  # 32
    parser.add_argument('--mu', default=0, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--expand-labels', default='False', help='expand labels to fit eval steps')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument("--power", default=0.9, type=float, help="power for learning rate decay")
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')

    # misc
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers')
    parser.add_argument('--seed', default=1388, type=int, help='random seed')
    parser.add_argument('--debug', default='True', help='do debug operation')

    # params set-up
    args = proj.project_args_setup(parser.parse_args(), params)
    return args


if __name__ == '__main__':
    dataset = "CIFAR10"
    main('E200_BS64_MHE_fd0_20240606.1', {'dataset': dataset, 'batch_size': 64, 'noisy_factor': 0})
