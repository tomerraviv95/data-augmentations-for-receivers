from python_code.detectors.deepsic.online_deep_sic_trainer import OnlineDeepSICTrainer
from python_code.detectors.vnet.vnet_trainer import VNETTrainer
from python_code.utils.config_singleton import Config

conf = Config()

if __name__ == '__main__':
    if conf.detector_type == 'deepsic':
        trainer = OnlineDeepSICTrainer()
    else:
        trainer = VNETTrainer()
    trainer.evaluate()
