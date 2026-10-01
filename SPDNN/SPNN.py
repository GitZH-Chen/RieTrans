import hydra
from omegaconf import DictConfig

from spnn.training.train import train

class Args:
    """ a Struct Class  """
    pass
args=Args()
args.config_name='SPNN.yaml'
@hydra.main(config_path='./conf/SPNN/', config_name=args.config_name, version_base='1.1')
def main(cfg: DictConfig):
    train(cfg,args)

if __name__ == "__main__":
    main()
