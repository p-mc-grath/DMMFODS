import logging

# TODO training loop
# TODO add optimizer to model
# TODO save chckpoints
# TODO Unpack training_0000.tar?
# Source: https://github.com/moemen95/Pytorch-Project-Template/blob/master/agents/condensenet.py
class Dense_U_Net_lidar_Agent():
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def run(self):
        try:
            if self.mode == 'test':
                self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()

            valid_acc = self.validate()
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        pass

    def validate(self):
        pass

