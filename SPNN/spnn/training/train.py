from torch.utils.tensorboard import SummaryWriter
import os
import logging
import json
import torch as th
import torch.nn as nn
import numpy as np
import fcntl

from spnn.utils import get_model
from spnn.utils.experiment import get_dataset_settings,optimzer,parse_cfg,train_per_epoch,val_per_epoch
import spnn.utils.experiment as experiment_utils
from spnn.utils.common_utils import set_seed_thread

def train(cfg,args):
    args = parse_cfg(args, cfg)
    args.folds=cfg.fit.folds

    # set logger
    logger = logging.getLogger(args.modelname)
    logger.setLevel(logging.INFO)
    args.logger = logger
    logger.info('begin model {} on dataset: {}'.format(args.modelname, args.dataset))

    # set seed and threadnum
    set_seed_thread(args.seed, args.threadnum)

    # begin K fold experiments
    acc_val_all = [];acc_val_last_k = []
    all_training_times = []
    for ith in range(args.folds):
        # set dataset, model and optimizer
        model = get_model.get_model(args)
        model.to(th.double)
        model.to(args.device)
        if ith==0:
            logger.info(model)
            if args.debug:
                    return

        args.DataLoader = get_dataset_settings(args)
        loss_fn = nn.CrossEntropyLoss()
        args.loss_fn = loss_fn.to(args.device)
        args.opti = optimzer(model.parameters(), lr=args.lr, mode=args.optimizer_mode, weight_decay=args.weight_decay,momentum=args.optimizer_momentum)
        if ith==0:
            logger.info(args.opti)

        # begin training
        args.ith_fold = ith+1
        logger.info(f'{args.ith_fold:d}/{args.folds:d} folds begins')
        # begin training
        val_acc, training_time = training_loop_of_KFold(model, args)
        all_training_times.extend(training_time)

        range_acc=int(10)
        acc_val_all.append(val_acc)
        last_k = np.array(val_acc)[-range_acc:]
        acc_val_last_k.append(last_k.max())

    save_print_results(args,acc_val_all,acc_val_last_k,logger,range_acc)
    if all_training_times:
        all_folds_times = np.asarray(all_training_times)
        all_folds_avg = all_folds_times.mean()
        all_folds_std = all_folds_times.std()
        logger.info(f"Fit time (all folds): {all_folds_avg:.4f}s ± {all_folds_std:.4f}s")
        save_time(args, all_folds_avg, all_folds_std)
    return acc_val_all

def training_loop_of_KFold(model, args):
    #setting tensorboard
    if args.is_writer:
        args.writer_path = os.path.join('./tensorboard_logs/', f"{args.modelname}_{args.ith_fold}")
        args.logger.info('writer path {}'.format(args.writer_path))
        args.writer = SummaryWriter(args.writer_path)

    acc_val = [];loss_val = [];acc_train = [];loss_train = [];training_time=[]
    logger = args.logger
    # training loop
    for epoch in range(0, args.epochs):
        # training
        elapse, epoch_loss_train, epoch_acc_train = train_per_epoch(model, args)
        training_time.append(elapse)
        acc_train.append(np.asarray(epoch_acc_train).mean() * 100)
        loss_train.append(np.asarray(epoch_loss_train).mean())

        # validation
        epoch_loss_val, epoch_acc_val = val_per_epoch(model, args)
        loss_val.append(np.asarray(epoch_loss_val).mean())
        acc_val.append(np.asarray(epoch_acc_val).mean() * 100)

        # save data into tensorboard
        if args.is_writer:
            args.writer.add_scalar('Loss/val', loss_val[epoch], epoch)
            args.writer.add_scalar('Accuracy/val', acc_val[epoch], epoch)
            args.writer.add_scalar('Loss/train', loss_train[epoch], epoch)
            args.writer.add_scalar('Accuracy/train', acc_train[epoch], epoch)

        # print results
        experiment_utils.print_results(logger, training_time, acc_val, loss_val, epoch, args)
    logger.info(
        'Fold {}/{}: best val acc in the last 10 epochs: {:.2f}% with avg time over the last 5 epochs: {:.2f}s'.format(
            args.ith_fold, args.folds, np.asarray(acc_val[-10:]).max(), np.asarray(training_time[-5:]).mean()))

    if args.is_writer:
        args.writer.close()
    return acc_val, training_time

def write_final_results(file_path,message):
    # Create a file lock
    with open(file_path, "a",encoding='utf-8') as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Acquire an exclusive lock

        # Write the message to the file
        file.write(message + "\n")

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
def save_print_results(args,acc_val_all,acc_val_last_k,logger,range):

    mean = np.asarray(acc_val_last_k).mean()
    std = np.asarray(acc_val_last_k).std()
    final_results = f'{args.folds} folds best-in-last-{range:d} average result is: {mean:.2f}±{std:.2f}'
    logger.info(final_results)

    final_results_path = os.path.join(os.getcwd(), 'final_results_' + args.dataset)
    logger.info("results file path: {}, and saving the results".format(final_results_path))
    write_final_results(final_results_path, args.modelname + '- ' + final_results)
    th.save({
        'acc_val_all': acc_val_all,
    }, os.path.join(os.getcwd(), f'torch_results_{args.modelname}.pt'))

def save_time(args, fit_time_avg, fit_time_std):
    file_path = os.path.join(os.getcwd(), f'time_{args.dataset}-{args.modelname}.jsonl')
    record = {
        "dataset": args.dataset,
        "modelname": args.modelname,
        "fit_time_all_folds_avg": float(f"{fit_time_avg:.6f}"),
        "fit_time_all_folds_std": float(f"{fit_time_std:.6f}")
    }
    with open(file_path, "a", encoding='utf-8') as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
