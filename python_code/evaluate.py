from python_code.detectors.deepsic.online_deep_sic_trainer import OnlineDeepSICTrainer
from python_code.detectors.vnet.vnet_trainer import VNETTrainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes

conf = Config()

CHANNEL_TYPE_TO_TRAINER_DICT = {ChannelModes.SISO.name: VNETTrainer,
                                ChannelModes.MIMO.name: OnlineDeepSICTrainer}

if __name__ == '__main__':
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[conf.channel_type]()
    trainer.evaluate()
