from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import fcntl
import socket
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import optimizers
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics,set_seed,parse_cfg,write_final_results


def train(cfg,args):
    args = parse_cfg(args, cfg)
    set_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    # set logger
    logger = logging.getLogger(args.modelname)
    logger.setLevel(logging.INFO)

    args.logger = logger
    logger.info('Dataset: {}'.format(args.modelname, args.dataset))
    logger.info(f'Using: {args.device}')
    logger.info("Using seed {}.".format(args.seed))

    fold_metrics=[]
    fold_val_metrics=[]
    all_training_times = []
    run_param_num = None
    for ith_fold in range(args.folds):
        ith_best_test_metrics, ith_best_val_metrics, ith_training_times, ith_param_num = train_process(args,ith_fold,logger)
        fold_metrics.append(ith_best_test_metrics)
        fold_val_metrics.append(ith_best_val_metrics)
        all_training_times.extend(ith_training_times)
        if run_param_num is None:
            run_param_num = ith_param_num

    # calculate the average metrics
    loss_values = [metrics['loss'].item() for metrics in fold_metrics]
    roc_values = [metrics['roc'] for metrics in fold_metrics]
    ap_values = [metrics['ap'] for metrics in fold_metrics]

    # Compute the mean and standard deviation
    average_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)
    average_roc = np.mean(roc_values)
    std_roc = np.std(roc_values)
    average_ap = np.mean(ap_values)
    std_ap = np.std(ap_values)

    message = f"{args.folds}-folds: " \
              f"average best loss: {average_loss * 100:.2f} ± {std_loss * 100:.2f}, " \
              f"average best ROC: {average_roc*100:.2f} ± {std_roc*100:.2f}, " \
              f"average best AP: {average_ap*100:.2f} ± {std_ap*100:.2f}"
    logger.info(message)
    write_final_results(args.dataset,args.modelname+' '+message)

    if all_training_times:
        all_best5_times = np.asarray(sorted(all_training_times)[:5])
        all_best5_avg = all_best5_times.mean()
        all_best5_std = all_best5_times.std()
        logger.info(
            f"Fit time (best 5 epochs, all folds): {all_best5_avg:.4f}s ± {all_best5_std:.4f}s; "
            f"param number: {run_param_num}"
        )
        save_time_param(args, all_best5_avg, all_best5_std, run_param_num)

    if args.save_results:
        tensor = torch.tensor(roc_values)
        folder_name = 'tensor_results'
        # Create the folder if it does not exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the tensor in the folder
        file_path = os.path.join(folder_name, f'{args.modelname}-roc_values.pt')
        torch.save(tensor, file_path)

        print(f"Tensor saved at: {file_path}")

    if args.save_tuning_metrics:
        val_roc_values = [float(metrics['roc']) for metrics in fold_val_metrics]
        val_ap_values = [float(metrics['ap']) for metrics in fold_val_metrics]
        test_roc_values = [float(metrics['roc']) for metrics in fold_metrics]
        test_ap_values = [float(metrics['ap']) for metrics in fold_metrics]
        tuning_record = {
            "dataset": args.dataset,
            "modelname": args.modelname,
            "folds": int(args.folds),
            "seed": int(args.seed),
            "manifold": args.manifold,
            "transform_mode": args.transform_mode,
            "normalize_v": bool(args.normalize_v),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "fold_best_val_roc": val_roc_values,
            "fold_best_val_ap": val_ap_values,
            "mean_best_val_roc": float(np.mean(val_roc_values)),
            "population_std_best_val_roc": float(np.std(val_roc_values)),
            "fold_test_roc_at_best_val": test_roc_values,
            "fold_test_ap_at_best_val": test_ap_values,
            "mean_test_roc_at_best_val": float(np.mean(test_roc_values)),
            "population_std_test_roc_at_best_val": float(np.std(test_roc_values)),
            "selection_metric": "mean_best_val_roc",
            "std_ddof": 0,
        }
        with open("tuning_metrics.json", "w", encoding="utf-8") as file:
            json.dump(tuning_record, file, indent=2, sort_keys=True)
            file.write("\n")
        logger.info("Saved raw tuning metrics to tuning_metrics.json")

def train_process(args,ith_fold,logger):
    # setting writer
    if args.is_writer:
        # if args.folds>1:
        #     args.writer_path = os.path.join('./tensorboard_logs', f"{args.modelname}_{args.ith_fold}")
        # else:
        #     args.writer_path = os.path.join('./tensorboard_logs/', f"{args.modelname}_{args.ith_fold}")
        args.writer_path = os.path.join('./tensorboard_logs/', f"{args.modelname}_{ith_fold+1}")
        logger.info('writer path {}'.format(args.writer_path))
        args.writer = SummaryWriter(args.writer_path)

    # Load data and model
    data = load_data(args, os.path.join(args.path, args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    input_n_nodes = int(args.n_nodes)
    input_feat_dim = int(args.feat_dim)
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logger.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logger.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(
        params=model.parameters(),
        lr=args.lr,
        amsgrad=args.amsgrad,
        weight_decay=args.weight_decay,
    )
    logger.info(f'Optimizer: {optimizer.__class__.__module__}.{optimizer.__class__.__name__}')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    measure_peak_memory = bool(args.measure_peak_memory)
    if measure_peak_memory:
        if args.device == 'cpu':
            raise RuntimeError('measure_peak_memory requires CUDA')
        if int(args.folds) != 1:
            raise RuntimeError('measure_peak_memory requires training.folds=1')
        torch.cuda.synchronize(args.device)
        baseline_allocated = int(torch.cuda.memory_allocated(args.device))
        baseline_reserved = int(torch.cuda.memory_reserved(args.device))
        torch.cuda.reset_peak_memory_stats(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    training_time = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        if measure_peak_memory and epoch == 0:
            torch.cuda.synchronize(args.device)
            device_index = torch.device(args.device).index
            device_properties = torch.cuda.get_device_properties(device_index)
            bytes_per_mib = 1024 ** 2
            memory_record = {
                'status': 'Complete',
                'measurement_boundary': (
                    'baseline after model, full-batch data, and optimizer construction; '
                    'peak covers encode, train decoder/loss, backward, and first optimizer step'
                ),
                'peak_metric_for_table': 'peak_allocated_mib',
                'dataset': args.dataset,
                'modelname': args.modelname,
                'transform_mode': args.transform_mode,
                'manifold': args.manifold,
                'normalize_v': bool(args.normalize_v),
                'seed': int(args.seed),
                'folds': int(args.folds),
                'measured_epoch': 1,
                'full_batch': True,
                'n_nodes': input_n_nodes,
                'feature_dim': input_feat_dim,
                'train_edges': int(args.nb_edges),
                'train_false_edge_pool': int(args.nb_false_edges),
                'dtype': str(torch.get_default_dtype()),
                'optimizer': optimizer.__class__.__name__,
                'learning_rate': float(args.lr),
                'weight_decay': float(args.weight_decay),
                'dropout': float(args.dropout),
                'param_num': int(tot_params),
                'baseline_allocated_bytes': baseline_allocated,
                'baseline_reserved_bytes': baseline_reserved,
                'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(args.device)),
                'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(args.device)),
                'baseline_allocated_mib': baseline_allocated / bytes_per_mib,
                'baseline_reserved_mib': baseline_reserved / bytes_per_mib,
                'peak_allocated_mib': torch.cuda.max_memory_allocated(args.device) / bytes_per_mib,
                'peak_reserved_mib': torch.cuda.max_memory_reserved(args.device) / bytes_per_mib,
                'hostname': socket.gethostname(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'gpu_name': device_properties.name,
                'gpu_total_memory_bytes': int(device_properties.total_memory),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'torch_version': torch.__version__,
                'torch_cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'torch_num_threads': torch.get_num_threads(),
                'thread_environment': {
                    name: os.environ.get(name)
                    for name in (
                        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'NUMEXPR_NUM_THREADS'
                    )
                },
            }
            with open('memory_metrics.json', 'w', encoding='utf-8') as file:
                json.dump(memory_record, file, indent=2, sort_keys=True)
                file.write('\n')
            logger.info(
                'Peak memory: allocated %.2f MiB, reserved %.2f MiB; baseline allocated %.2f MiB',
                memory_record['peak_allocated_mib'],
                memory_record['peak_reserved_mib'],
                memory_record['baseline_allocated_mib'],
            )
        lr_scheduler.step()
        training_time.append(time.time() - t)
        if (epoch + 1) % args.log_freq == 0:
            logger.info(" ".join([f'Train: Fold: {ith_fold+1}/{args.folds}, Epoch:{epoch + 1}/{args.epochs}',
                                  'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                  format_metrics(train_metrics),
                                  'fit time: {:.4f}s'.format(training_time[-1])
                                  ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                embeddings = model.encode(data['features'], data['adj_train_norm'])
                val_metrics = model.compute_metrics(embeddings, data, 'val')
                if (epoch + 1) % args.log_freq == 0:
                    logger.info(" ".join([f'Val: Fold: {ith_fold+1}/{args.folds}, Epoch:{epoch + 1}/{args.epochs}', format_metrics(val_metrics)]))
                if model.has_improved(best_val_metrics, val_metrics):
                    best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                    best_emb = embeddings.cpu()
                    if args.save:
                        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                    best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epochs:
                        logger.info("Early stopping")
                        break

        # save data into tensorboard
        if args.is_writer:
            args.writer.add_scalar('loss/val', val_metrics['loss'].item(), epoch)
            args.writer.add_scalar('roc/val', val_metrics['roc']*100, epoch)
            args.writer.add_scalar('ap/val', val_metrics['ap']*100, epoch)
            args.writer.add_scalar('loss/train', train_metrics['loss'].item(), epoch)
            args.writer.add_scalar('roc/train', train_metrics['roc']*100, epoch)
            args.writer.add_scalar('ap/train', train_metrics['ap']*100, epoch)
            args.writer.add_scalar('loss/best_test', best_test_metrics['loss'].item(), epoch)
            args.writer.add_scalar('roc/best_test', best_test_metrics['roc']*100, epoch)
            args.writer.add_scalar('ap/best_test', best_test_metrics['ap']*100, epoch)

    logger.info("Optimization Finished!")
    lastk_average_time = np.asarray(training_time[-5:]).mean()
    best5_times = np.asarray(sorted(training_time)[:5])
    best5_average_time = best5_times.mean()
    best5_std_time = best5_times.std()
    logger.info(f"Total time elapsed: {time.time() - t_total:.4f}s "
                f"with average time: {lastk_average_time:.4f} "
                f"and average smallest time: {best5_average_time:.4f}")
    if not best_test_metrics:
        model.eval()
        with torch.no_grad():
            best_emb = model.encode(data['features'], data['adj_train_norm'])
            best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logger.info(" ".join(["Val set results:", format_metrics(best_val_metrics)]))
    logger.info(" ".join(["Test set results:", format_metrics(best_test_metrics)]))
    logger.info(
        f"Fit time (best 5 epochs): {best5_average_time:.4f}s ± {best5_std_time:.4f}s; "
        f"param number: {tot_params}"
    )
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logger.info(f"Saved model in {save_dir}")

    return best_test_metrics, best_val_metrics, training_time, tot_params

def save_time_param(args, fit_time_avg, fit_time_std, param_num):
    folder = 'time_param'
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, f'{args.dataset}-{args.modelname}')
    record = {
        "dataset": args.dataset,
        "modelname": args.modelname,
        "fit_time_best5_avg": float(f"{fit_time_avg:.6f}"),
        "fit_time_best5_std": float(f"{fit_time_std:.6f}"),
        "param_num": int(param_num),
    }
    with open(file_path, "a", encoding='utf-8') as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
