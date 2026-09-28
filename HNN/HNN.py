import hydra
from omegaconf import DictConfig

from utils.train import train

class Args:
    """ a Struct Class  """
    pass
args=Args()
args.config_name='HNNs.yaml'


@hydra.main(config_path='./conf/HNN/', config_name=args.config_name, version_base='1.1')
def main(cfg: DictConfig):
    train(cfg,args)

if __name__ == '__main__':
    main()
