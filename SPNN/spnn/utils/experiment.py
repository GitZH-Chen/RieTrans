import datetime
import geoopt
import time
import torch as th
from omegaconf import DictConfig, OmegaConf


from datasets.baselines.FPHA_Loader import DataLoaderFPHA

from datasets.spnn.FPHA_TPR import DataLoaderFPHA_TPR
from datasets.spnn.Radar import DataLoaderRadar

from geometry.spd.spd_matrices import tril_param_metrics,bi_param_metrics,single_param_metrics

def get_dataset_settings(args):
    if args.dataset=='RADAR':
        DataLoader = DataLoaderRadar(
            args.path,
            pval=0.25,
            ptest=0.25,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    elif args.dataset=='FPHA':
        DataLoader = DataLoaderFPHA(args.path,args.batch_size)
    elif args.dataset=='FPHA_TPR':
        DataLoader = DataLoaderFPHA_TPR(args.path,args.batch_size)
    else:
        raise Exception('unknown dataset {}'.format(args.dataset))
    return DataLoader

def get_model_name(args):
    if args.model_type == 'GyroSPD':
        optim = f'{args.optimizer_mode}-{args.lr}-m_{args.optimizer_momentum}-{args.weight_decay}'
        name = f'{args.batch_size}-{args.seed}-{args.model_type}-{args.architecture}-{args.metric}-{optim}-{datetime.datetime.now().strftime("%H_%M")}'
    elif args.model_type == 'GyroSPDpp':
        optim = f'{args.optimizer_mode}-{args.lr}-m_{args.optimizer_momentum}-{args.weight_decay}'
        name = (
            f'{args.batch_size}-{args.seed}-{args.model_type}-{args.architecture}-{args.channels}-'
            f'{args.conv_metric}-{args.clf_metric}-{optim}-{datetime.datetime.now().strftime("%H_%M")}'
        )
    elif args.model_type == 'SPNN' or args.model_type.startswith('SPD'):
        description=''
        if args.model_type in ['SPNN', 'SPDNN']:
            if args.conv_metric == 'PEM' and args.clf_metric == 'PEM':
                args.clf_power=args.conv_power
        if args.classifier == 'SPDMLR':
            if args.clf_metric in tril_param_metrics:
                description = f'{args.clf_metric}-[{args.clf_power}]'
            elif args.clf_metric in bi_param_metrics:
                description = args.clf_metric
            elif args.clf_metric in single_param_metrics:
                description = f'{args.clf_metric}-[{args.clf_power}]'
            phi = 'Y' if args.clf_is_phi else 'N'
            description = f'-{description}-phi_{phi}'

        if args.model_type == 'SPDResNet':
            description = description + f'-{args.ResBlockMetric}'
        elif args.model_type == 'SPDNetRBN':
            if args.BN_type=='brooks':
                description = description + f'-{args.BN_type}-m_{args.momentum}'
            elif args.BN_type == 'LieBN':
                description = description + f'-{args.BN_type}-m_{args.momentum}-{args.bn_metric}-[{args.bn_power}]'
        if args.model_type in ['SPNN', 'SPDNN']:
            transform_mode = f'{args.transform_mode}-{args.architecture}-{args.channels}'
        else:
            transform_mode = f'{args.transform_mode}-{args.architecture}'

        if args.transform_mode == 'SPDLinear':
            phiinv = 'Y' if args.conv_is_phi_inv else 'N'
            transform_mode = f'{transform_mode}-{args.conv_metric}-[{args.conv_power:.2f}]-phiinv_{phiinv}'
            if args.act_power:
                transform_mode = f'[{args.act_power}]-{transform_mode}'
        optim = f'{args.optimizer_mode}-{args.lr}-m_{args.optimizer_momentum}-{args.weight_decay}'
        name = f'{args.batch_size}-{args.seed}-{args.model_type}-{transform_mode}-{args.classifier}{description}-{optim}-{datetime.datetime.now().strftime("%H_%M")}'
    return name

def optimzer(parameters,lr,mode='AMSGRAD',weight_decay=0.,momentum=0):
    if mode=='SGD':
        optim = geoopt.optim.RiemannianSGD(parameters, lr=lr,weight_decay=weight_decay,momentum=momentum)
    elif mode=='AMSGRAD':
        optim = geoopt.optim.RiemannianAdam(parameters, lr=lr,amsgrad=True,weight_decay=weight_decay)
    else:
        raise Exception('unknown optimizer {}'.format(mode))
    return optim

def parse_cfg(args, cfg: DictConfig):
    # Function to recursively set attributes, keeping only the final key name
    def set_attributes_from_dict(target, source):
        for key, value in source.items():
            if isinstance(value, dict):
                # If the value is a dict, continue to extract its values
                set_attributes_from_dict(target, value)
            else:
                # Directly set the attribute on the target
                setattr(target, key, value)

    # Convert Hydra config to a nested dictionary and then flatten it
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    set_attributes_from_dict(args, cfg_dict)

    args.device = "cpu" if cfg.fit.device == 'cpu' else th.device(f"cuda:{cfg.fit.device}")

    # get model name
    args.modelname = get_model_name(args)
    return args

def train_per_epoch(model,args):
    start = time.time()
    epoch_loss, epoch_acc = [], []
    model.train()
    for local_batch, local_labels in args.DataLoader._train_generator:
        local_batch = local_batch.double().to(args.device)
        local_labels = local_labels.to(args.device)
        args.opti.zero_grad()
        out = model(local_batch)
        l = args.loss_fn(out, local_labels)
        acc, loss = (out.argmax(1) == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
        epoch_loss.append(loss)
        epoch_acc.append(acc)
        l.backward()
        args.opti.step()
    end = time.time()
    elapse = end - start
    return elapse,epoch_loss,epoch_acc

def val_per_epoch(model,args):
    epoch_loss, epoch_acc = [], []
    y_true, y_pred = [], []
    model.eval()
    with th.no_grad():
        for local_batch, local_labels in args.DataLoader._test_generator:
            local_batch = local_batch.double().to(args.device)
            local_labels=local_labels.to(args.device)
            out = model(local_batch)
            l = args.loss_fn(out, local_labels)
            predicted_labels = out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy()))
            y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc, loss = (predicted_labels == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
            epoch_acc.append(acc)
            epoch_loss.append(loss)
    return epoch_loss,epoch_acc

def print_results(logger,training_time,acc_val,loss_val,epoch,args):
    if epoch % args.cycle == 0:
        logger.info(f'Time: {training_time[epoch]:.2f}, Val acc: {acc_val[epoch]:.2f}, loss: {loss_val[epoch]:.2f} at epoch {epoch + 1:d}/{args.epochs:d}')
