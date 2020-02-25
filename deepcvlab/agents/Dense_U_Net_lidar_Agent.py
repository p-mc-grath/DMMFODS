import os
import logging
import numpy as np
import torch
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ..graphs.models.Dense_U_Net_lidar import densenet121_u_lidar, Dense_U_Net_lidar
from ..utils import Dense_U_Net_lidar_helper
from ..datasets.WaymoData import WaymoDataset_Loader

# optimizes performance if input size same at each iteration
cudnn.benchmark = True

# Customized version of: https://github.com/moemen95/Pytorch-Project-Template/blob/master/agents/condensenet.py 

class Dense_U_Net_lidar_Agent:
    def __init__(self, config=None, torchvision_init=True):
        '''
        Arguments:  
            torchvision_init: boolean
                - True:     load densenet state dict from torchvision
                - False:    load checkpoint
        '''
        self.logger = logging.getLogger("Agent")

        # model and config if lazy
        self.model = densenet121_u_lidar(pretrained=torchvision_init, 
            config=config)
        
        # in case config is empty it is created in model
        self.config = self.model.config

        # dataloader
        self.data_loader = WaymoDataset_Loader(self.config)

        # pixel-wise cross-entropy loss 
        self.loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.config.optimizer.learning_rate, 
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2), 
            eps=self.config.optimizer.eps, weight_decay=self.config.optimizer.weight_decay, 
            amsgrad=self.config.optimizer.amsgrad)

        # learning rate decay scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
            step_size=self.config.optimizer.weight_decay.every_n_epochs, 
            gamma=self.config.optimizer.weight_decay.gamma)

        # initialize counters; updated in load_checkpoint
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_val_acc = 0

        # if cuda is available export model to gpu
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config.agent.seed)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.agent.seed)
            self.logger.info("Operation will be on *****CPU***** ")
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        if not torchvision_init:
            self.load_checkpoint(self.config.dir.pretrained_weights.best_checkpoint)

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.dir.root, comment='Dense_U_Net')
    
    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            self.config.agent.checkpoint.epoch: self.current_epoch,
            self.config.agent.checkpoint.iteration: self.current_iteration,
            self.config.agent.checkpoint.best_val_acc: self.best_val_acc,
            self.config.agent.checkpoint.state_dict: self.model.state_dict(),
            self.config.agent.checkpoint.optimizer: self.optimizer.state_dict()
        }
        # Save the state
        if is_best:
            torch.save(state, self.config.dir.pretrained_weights.best_checkpoint)
        else:
            torch.save(state, os.path.join(self.config.dir.pretrained_weights, filename))
    
    def load_checkpoint(self, filename):
        '''
        load checkpoint from file
        should contain following keys: 
            'epoch', 'iteration', 'best_val_acc', 'state_dict', 'optimizer'
            where state_dict is model statedict
            and optimizer is optimizer statesict
        '''
        filename = os.path.join(self.config.dir.pretrained_weights, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint[self.config.agent.checkpoint.epoch]
            self.current_iteration = checkpoint[
                self.config.agent.checkpoint.iteration]
            self.best_val_acc = checkpoint[
                self.config.agent.checkpoint.best_val_acc]
            self.model.load_state_dict(checkpoint[
                self.config.agent.checkpoint.state_dict])
            self.optimizer.load_state_dict(checkpoint[
                self.config.agent.checkpoint.optimizer])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        '''
        can handle keyboard interupt
        '''
        try:
            if self.config.loader.mode == 'test':
                self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        '''
        train starting from checkpoint/ 0
        save checkpoints after each epoch
        save best checkpoints in separate file
        '''
        for epoch in range(self.current_epoch, self.config.agent.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()

            val_acc = self.validate()
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize progress visualization and get batch
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))
        
        # Set the model to be in training mode
        self.model.train()

        current_batch = 0
        epoch_loss = [0]
        for image, lidar, _, ht_map in tqdm_batch:
            # push to gpu if possible
            if self.cuda:
                image = image.cuda(non_blocking=self.config.loader.async_loading)
                lidar = lidar.cuda(non_blocking=self.config.loader.async_loading)
                ht_map = ht_map.cuda(non_blocking=self.config.loader.async_loading)

            # TODO rmv: debugging
            print(image.shape)
            print(lidar.shape)
            print(ht_map.shape)

            # forward pass
            prediction = self.model(image, lidar)
            
            # pixel-wise loss
            current_loss = self.loss(prediction, ht_map)
            if np.isnan(float(current_loss.item())):
                raise ValueError('Loss is nan during training...')
            epoch_loss += [epoch_loss[-1] + current_loss.item()]

            # backprop
            self.optimizer.zero_grad()
            current_loss.backward()
            self.optimizer.step()

            # counters
            self.current_iteration += 1
            current_batch += 1

            # log
            self.summary_writer.add_scalar("loss/iteration", current_loss.item(), self.current_iteration)

        tqdm_batch.close()

        # learning rate decay update; after validate; after each epoch
        self.lr_scheduler.step()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "average loss: " + str(
             epoch_loss[-1]/len(epoch_loss)))

    def validate(self):
        """
        One epoch validation
        :return:
        """
        # Initialize progress visualization and get batch
        # !self.data_loader.valid_loader works for both valid and test 
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = [0]
        for image, lidar, _, ht_map in tqdm_batch:
            # push to gpu if possible
            if self.cuda:
                image = image.cuda(non_blocking=self.config.loader.async_loading)
                lidar = lidar.cuda(non_blocking=self.config.loader.async_loading)
                ht_map = ht_map.cuda(non_blocking=self.config.loader.async_loading)

            # forward pass
            prediction = self.model(image, lidar)
            
            # pixel-wise loss
            current_loss = self.loss(prediction, ht_map)
            if np.isnan(float(current_loss.item())):
                raise ValueError('Loss is nan during training...')
            epoch_loss += [epoch_loss[-1] + current_loss.item()]

        avg_val_acc = epoch_loss[-1]/len(epoch_loss) #
        self.logger.info("Validation results at epoch-" + str(self.current_epoch) + " | " + "average loss: " + str(
            avg_val_acc))

        tqdm_batch.close()
        
        return avg_val_acc

    def finalize(self):
        """
        Save checkpoint and log
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.dir.root))
        self.summary_writer.close()
