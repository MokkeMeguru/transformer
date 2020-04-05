import hydra

@hydra.main(config_path='./conf/config.yaml')
def my_app(cfg):
    print(cfg.pretty())

if __name__ == "__main__":
    my_app()
