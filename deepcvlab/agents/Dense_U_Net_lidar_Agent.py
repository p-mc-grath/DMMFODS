import os
import logging
import torch
import warnings
import numpy as np
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from ..graphs.models.Dense_U_Net_lidar import densenet121_u_lidar, Dense_U_Net_lidar
from ..utils import Dense_U_Net_lidar_helper as utils
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
                - False:    load checkpoint; if no checkpoint just normal init
        '''
        self.logger = logging.getLogger("Agent")

        # model and config if lazy
        self.model = densenet121_u_lidar(pretrained=torchvision_init, 
            config=config)
        
        # in case config is empty it is created in model
        self.config = self.model.config

        # dataloader
        print('Creation of Dataset started')
        self.data_loader = WaymoDataset_Loader(self.config)

        # pixel-wise cross-entropy loss 
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none').cuda()
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.config.optimizer.learning_rate, 
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2), 
            eps=self.config.optimizer.eps, weight_decay=self.config.optimizer.weight_decay, 
            amsgrad=self.config.optimizer.amsgrad)

        # learning rate decay scheduler
        if self.config.optimizer.lr_scheduler.want:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size=self.config.optimizer.lr_scheduler.every_n_epochs, 
                gamma=self.config.optimizer.lr_scheduler.gamma)

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
            self.load_checkpoint()

        # Tensorboard Writer
        Path(self.config.dir.summary).mkdir(exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=self.config.dir.summary, comment='Dense_U_Net')
    
    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=False):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        #aggregate important data
        state = {
            self.config.agent.checkpoint.epoch: self.current_epoch,
            self.config.agent.checkpoint.iteration: self.current_iteration,
            self.config.agent.checkpoint.best_val_acc: self.best_val_acc,
            self.config.agent.checkpoint.state_dict: self.model.state_dict(),
            self.config.agent.checkpoint.optimizer: self.optimizer.state_dict()
        }
        
        if is_best:
            filename = self.config.agent.best_checkpoint_name

        # create dir if not exists
        Path(self.config.dir.pretrained_weights).mkdir(exist_ok=True)

        # Save the state
        torch.save(state, os.path.join(self.config.dir.pretrained_weights, filename))
    
    def load_checkpoint(self, filename=None):
        '''
        load checkpoint from file
        should contain following keys: 
            'epoch', 'iteration', 'best_val_acc', 'state_dict', 'optimizer'
            where state_dict is model statedict
            and optimizer is optimizer statesict
        '''
        # use best if not specified
        if filename is None:
            filename = self.config.agent.best_checkpoint_name

        filepath = os.path.join(self.config.dir.pretrained_weights, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filepath)

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
                             .format(self.config.dir.pretrained_weights, checkpoint['epoch'], checkpoint['iteration']))
        except OSError:
            warnings.warn("No checkpoint exists from '{}'. Skipping...".format(filepath))
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(filepath))
            self.logger.info("**First time to train**")

    def run(self):
        '''
        can handle keyboard interupt
        '''
        print('starting ' + self.config.loader.mode + ' at ' + str(datetime.now()))
        try:
            if self.config.loader.mode == 'test':
                with torch.no_grad():
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

            with torch.no_grad():
                avg_val_acc_per_class = self.validate()
            val_acc = torch.mean(avg_val_acc_per_class)
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(is_best=is_best)

        self.summary_writer.close()

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
        epoch_loss = [torch.zeros(self.config.model.num_classes).to(self.device)]
        epoch_iou = [torch.zeros(self.config.model.num_classes)]
        epoch_iou_nans = [torch.zeros(self.config.model.num_classes)]
        epoch_acc = [torch.zeros(self.config.model.num_classes)]
        for image, lidar, ht_map in tqdm_batch:
            # push to gpu if possible
            if self.cuda:
                image = image.cuda(non_blocking=self.config.loader.async_loading)
                lidar = lidar.cuda(non_blocking=self.config.loader.async_loading)
                ht_map = ht_map.cuda(non_blocking=self.config.loader.async_loading)

            # forward pass
            prediction = self.model(image, lidar)
            
            # pixel-wise loss
            current_loss = self.loss(prediction, ht_map)
            loss_per_class = torch.sum(current_loss.detach(), dim=(0,2,3))
            epoch_loss += [epoch_loss[-1] + loss_per_class]

            # whole image IoU per class; not taking nans into acc for the mean value; counting the nans separately
            iou_per_instance_per_class = utils.compute_IoU_whole_img_batch(prediction.detach(), ht_map.detach(), self.config.agent.iou_threshold)
            iou_per_class = torch.tensor(np.nanmean(iou_per_instance_per_class, axis=0))
            iou_per_class[torch.isnan(iou_per_class)] = 0
            epoch_iou += [epoch_iou[-1] + iou_per_class]
            epoch_iou_nans += [epoch_iou_nans[-1] + torch.sum(torch.isnan(iou_per_instance_per_class), axis=0)]
            
            # compute class-wise accuracy of current batch
            acc_per_class = utils.compute_accuracy(ht_map.detach(), prediction.detach(), self.config.agent.iou_threshold)
            epoch_acc += [epoch_acc[-1] + acc_per_class]

            # backprop
            self.optimizer.zero_grad()
            current_loss.backward(torch.ones_like(current_loss.detach(), device=self.device))                            # , retain_graph=True?
            self.optimizer.step()

            # counters
            self.current_iteration += 1
            current_batch += 1

            # logging for visualization during training
            info_per_class_dict = {
                'loss vehicle': loss_per_class[0],
                'loss pedestrian': loss_per_class[1],
                'loss cyclist': loss_per_class[2],
                'acc vehicle': acc_per_class[0],
                'acc pedestrian': acc_per_class[1],
                'acc cyclist': acc_per_class[2],
                'iou vehicle': iou_per_class[0],
                'iou pedestrian': iou_per_class[1],
                'iou cyclist': iou_per_class[2],
                '#NaNs all': torch.sum(epoch_iou_nans)
            }
            self.summary_writer.add_scalars("Training_Info/", info_per_class_dict, self.current_iteration)

        tqdm_batch.close()

        # learning rate decay update; after validate; after each epoch
        if self.config.optimizer.lr_scheduler.want:
            self.lr_scheduler.step()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "average loss: " + str(
             epoch_loss[-1]/len(epoch_loss)) + " | " + "average IoU: " + str(epoch_iou[-1]/len(epoch_iou)) +
             ' | ' + '#NaNs V P C ' + str(epoch_iou_nans) + ' | ' + 'avg acc: ' + str(epoch_acc[-1]/len(epoch_acc)))

    def validate(self):
        """
        One epoch validation
        :return: 
            average acc per class
        """
        # Initialize progress visualization and get batch
        # !self.data_loader.valid_loader works for both valid and test 
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = [torch.zeros(self.config.model.num_classes).to(self.device)]
        epoch_iou = [torch.zeros(self.config.model.num_classes)]
        epoch_iou_nans = [torch.zeros(self.config.model.num_classes)]
        epoch_acc = [torch.zeros(self.config.model.num_classes)]
        for image, lidar, ht_map in tqdm_batch:
            # push to gpu if possible
            if self.cuda:
                image = image.cuda(non_blocking=self.config.loader.async_loading)
                lidar = lidar.cuda(non_blocking=self.config.loader.async_loading)
                ht_map = ht_map.cuda(non_blocking=self.config.loader.async_loading)

            # forward pass
            prediction = self.model(image, lidar)
            
            # pixel-wise loss
            current_loss = self.loss(prediction, ht_map)
            loss_per_class = torch.sum(current_loss.detach(), dim=(0,2,3))
            epoch_loss += [epoch_loss[-1] + loss_per_class]
            
            # whole image IoU per class; not taking nans into acc for the mean value; counting the nans separately
            iou_per_instance_per_class = utils.compute_IoU_whole_img_batch(prediction.detach(), ht_map.detach(), self.config.agent.iou_threshold)
            iou_per_class = torch.tensor(np.nanmean(iou_per_instance_per_class, axis=0))
            iou_per_class[torch.isnan(iou_per_class)] = 0
            epoch_iou += [epoch_iou[-1] + iou_per_class]
            epoch_iou_nans += [epoch_iou_nans[-1] + torch.sum(torch.isnan(iou_per_instance_per_class), axis=0)]

            # compute class-wise accuracy of current batch
            epoch_acc += [epoch_acc[-1] + 
                utils.compute_accuracy(ht_map.detach(), prediction.detach(), self.config.agent.iou_threshold)]

        self.logger.info("Validation at epoch-" + str(self.current_epoch) + " | " + "average loss: " + str(
             epoch_loss[-1]/len(epoch_loss)) + " | " + "average IoU: " + str(epoch_iou[-1]/len(epoch_iou)) +
             ' | ' + '#NaNs V P C ' + str(epoch_iou_nans) + ' | ' + 'avg acc: ' + str(epoch_acc[-1]/len(epoch_acc)))

        tqdm_batch.close()
        
        return epoch_acc[-1]/len(epoch_acc)

    def finalize(self):
        """
        Save checkpoint and log
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.summary_writer.close()
        print('ending ' + self.config.loader.mode + ' at ' + str(datetime.now()))
