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
        self.current_train_iteration = 0
        self.current_val_iteration = 0
        self.best_val_iou = 0

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

        # Tensorboard Writers 
        Path(self.config.dir.current_run.summary).mkdir(exist_ok=True, parents=True)
        self.train_summary_writer = SummaryWriter(log_dir=self.config.dir.current_run.summary, comment='Dense_U_Net')
        self.val_summary_writer = SummaryWriter(log_dir=self.config.dir.current_run.summary, comment='Dense_U_Net')

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=False):
        '''
        Saving the latest checkpoint of the training

        Arguments:
            filename: filename which will contain the state
            is_best: flag is it is the best model
        '''

        #aggregate important data
        state = {
            self.config.agent.checkpoint.epoch: self.current_epoch,
            self.config.agent.checkpoint.train_iteration: self.current_train_iteration,
            self.config.agent.checkpoint.val_iteration: self.current_val_iteration,
            self.config.agent.checkpoint.best_val_iou: self.best_val_iou,
            self.config.agent.checkpoint.state_dict: self.model.state_dict(),
            self.config.agent.checkpoint.optimizer: self.optimizer.state_dict()
        }
        
        if is_best:
            filename = self.config.agent.best_checkpoint_name

        # create dir if not exists
        Path(self.config.dir.current_run.checkpoints).mkdir(exist_ok=True, parents=True)

        # Save the state
        torch.save(state, os.path.join(self.config.dir.current_run.checkpoints, filename))
    
    def load_checkpoint(self, filename=None):
        '''
        load checkpoint from file
        should contain following keys: 
            'epoch', 'iteration', 'best_val_iou', 'state_dict', 'optimizer'
            where state_dict is model statedict
            and optimizer is optimizer statesict
        '''

        # use best if not specified
        if filename is None:
            filename = self.config.agent.best_checkpoint_name

        filepath = os.path.join(self.config.dir.current_run.checkpoints, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filepath)

            self.current_epoch = checkpoint[self.config.agent.checkpoint.epoch]
            self.current_train_iteration = checkpoint[
                self.config.agent.checkpoint.train_iteration]
            self.current_val_iteration = checkpoint[
                self.config.agent.checkpoint.val_iteration]
            self.best_val_iou = checkpoint[
                self.config.agent.checkpoint.best_val_iou]
            self.model.load_state_dict(checkpoint[
                self.config.agent.checkpoint.state_dict])
            self.optimizer.load_state_dict(checkpoint[
                self.config.agent.checkpoint.optimizer])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.dir.current_run.checkpoints, checkpoint['epoch'], checkpoint['train_iteration']))
        except OSError:
            warnings.warn("No checkpoint exists from '{}'. Skipping...".format(filepath))
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(filepath))
            self.logger.info("**First time to train**")

    def run(self):
        '''
        starts training are testing: specify under config.loader.mode
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
        training one epoch at a time
        validating after each epoch
        saving checkpoint after each epoch
        check if val acc is best and store separately
        '''

        # add selected loss and optimizer to config  | not added in init as may be changed before training
        self.config.loss.func = str(self.loss)
        self.config.optimizer.func = str(self.optimizer)

        # make sure to remember the hyper params
        self.add_hparams_summary_writer()
        self.save_hparams_json()

        # Iterate epochs | train one epoch | validate | save checkpoint
        for epoch in range(self.current_epoch, self.config.agent.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()

            with torch.no_grad():
                avg_val_iou_per_class = self.validate()

            val_iou = sum(avg_val_iou_per_class)/len(avg_val_iou_per_class)
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
            self.save_checkpoint(is_best=is_best)

        self.train_summary_writer.close()
        self.val_summary_writer.close()

    def train_one_epoch(self):
        '''
        One epoch training function
        '''

        # Initialize progress visualization and get batch
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))
        
        # Set the model to be in training mode
        self.model.train()

        current_batch = 0
        number_of_batches = self.data_loader.train_loader.dataset.__len__()
        epoch_loss = torch.zeros((number_of_batches, self.config.model.num_classes)).to(self.device)
        epoch_iou = torch.zeros((number_of_batches, self.config.model.num_classes))
        epoch_iou_nans = torch.zeros((number_of_batches, self.config.model.num_classes))
        epoch_acc = torch.zeros((number_of_batches, self.config.model.num_classes)).to(self.device)
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
            epoch_loss[current_batch, :] = loss_per_class

            # whole image IoU per class; not taking nans into acc for the mean value; counting the nans separately
            iou_per_instance_per_class = utils.compute_IoU_whole_img_batch(prediction.detach(), ht_map.detach(), self.config.agent.iou_threshold)
            iou_per_class = torch.tensor(np.nanmean(iou_per_instance_per_class, axis=0))
            iou_per_class[torch.isnan(iou_per_class)] = 0
            epoch_iou[current_batch, :] = iou_per_class
            epoch_iou_nans[current_batch, :] = torch.sum(torch.isnan(iou_per_instance_per_class), axis=0)
            
            # compute class-wise accuracy of current batch
            acc_per_class = utils.compute_accuracy(ht_map.detach(), prediction.detach(), self.config.agent.iou_threshold)
            epoch_acc[current_batch, :] = acc_per_class

            # backprop
            self.optimizer.zero_grad()
            current_loss.backward(torch.ones_like(current_loss.detach(), device=self.device))                            # , retain_graph=True?
            self.optimizer.step()

            # logging for visualization during training: separate plots for loss, acc, iou | each-classwise + overall
            loss_dict = {
                'Vehicle': loss_per_class[0],
                'Pedestrian': loss_per_class[1],
                'Cyclist': loss_per_class[2],
                'Overall': torch.mean(loss_per_class)
            }
            self.train_summary_writer.add_scalars("Training/Loss", loss_dict, self.current_train_iteration)
            acc_dict = {
                'Vehicle': acc_per_class[0],
                'Pedestrian': acc_per_class[1],
                'Cyclist': acc_per_class[2],
                'Overall': torch.mean(acc_per_class)
            }
            self.train_summary_writer.add_scalars("Training/Accuracy", acc_dict, self.current_train_iteration)
            iou_dict = {
                'Vehicle': iou_per_class[0],
                'Pedestrian': iou_per_class[1],
                'Cyclist': iou_per_class[2],
                'Overall': torch.mean(iou_per_class)
            }
            self.train_summary_writer.add_scalars("Training/IoU", iou_dict, self.current_train_iteration)

            # counters
            self.current_train_iteration += 1
            current_batch += 1

        tqdm_batch.close()

        # learning rate decay update; after validate; after each epoch
        if self.config.optimizer.lr_scheduler.want:
            self.lr_scheduler.step()

        avg_epoch_loss = torch.mean(epoch_loss, axis=0).tolist()
        avg_epoch_iou = torch.mean(epoch_iou, axis=0).tolist()
        cum_epoch_nans = torch.sum(epoch_iou_nans, axis=0).tolist()
        avg_epoch_acc = torch.mean(epoch_acc, axis=0).tolist()
        
        self.logger.info("Training at Epoch-" + str(self.current_epoch) + " | " + "Average Loss: " + str(
             avg_epoch_loss) + " | " + "Average IoU: " + str(avg_epoch_iou) + ' | ' + 'Number of NaNs: ' + str(
                 cum_epoch_nans) + ' | ' + 'Average Accuracy: ' + str(avg_epoch_acc))

    def validate(self):
        '''
        One epoch validation
        
        return: 
            average acc per class
        '''

        # Initialize progress visualization and get batch
        # !self.data_loader.valid_loader works for both valid and test 
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        current_batch = 0
        number_of_batches = self.data_loader.valid_loader.dataset.__len__()
        epoch_loss = torch.zeros((number_of_batches, self.config.model.num_classes)).to(self.device)
        epoch_iou = torch.zeros((number_of_batches, self.config.model.num_classes))
        epoch_iou_nans = torch.zeros((number_of_batches, self.config.model.num_classes))
        epoch_acc = torch.zeros((number_of_batches, self.config.model.num_classes)).to(self.device)
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
            epoch_loss[current_batch, :] = loss_per_class
            
            # whole image IoU per class; not taking nans into acc for the mean value; counting the nans separately
            iou_per_instance_per_class = utils.compute_IoU_whole_img_batch(prediction.detach(), ht_map.detach(), self.config.agent.iou_threshold)
            iou_per_class = torch.tensor(np.nanmean(iou_per_instance_per_class, axis=0))
            iou_per_class[torch.isnan(iou_per_class)] = 0
            epoch_iou[current_batch, :] = iou_per_class
            epoch_iou_nans[current_batch, :] = torch.sum(torch.isnan(iou_per_instance_per_class), axis=0)

            # compute class-wise accuracy of current batch
            acc_per_class = utils.compute_accuracy(ht_map.detach(), prediction.detach(), self.config.agent.iou_threshold)
            epoch_acc[current_batch, :] = acc_per_class

            # logging for visualization during training: separate plots for loss, acc, iou | each-classwise + overall
            loss_dict = {
                'Vehicle': loss_per_class[0],
                'Pedestrian': loss_per_class[1],
                'Cyclist': loss_per_class[2],
                'Overall': torch.mean(loss_per_class)
            }
            self.val_summary_writer.add_scalars("Validation/Loss", loss_dict, self.current_val_iteration)
            acc_dict = {
                'Vehicle': acc_per_class[0],
                'Pedestrian': acc_per_class[1],
                'Cyclist': acc_per_class[2],
                'Overall': torch.mean(acc_per_class)
            }
            self.val_summary_writer.add_scalars("Validation/Accuracy", acc_dict, self.current_val_iteration)
            iou_dict = {
                'Vehicle': iou_per_class[0],
                'Pedestrian': iou_per_class[1],
                'Cyclist': iou_per_class[2],
                'Overall': torch.mean(iou_per_class)
            }
            self.val_summary_writer.add_scalars("Validation/IoU", iou_dict, self.current_val_iteration)

            # counters
            self.current_val_iteration += 1
            current_batch += 1

        avg_epoch_loss = torch.mean(epoch_loss, axis=0).tolist()
        avg_epoch_iou = torch.mean(epoch_iou, axis=0).tolist()
        cum_epoch_nans = torch.sum(epoch_iou_nans, axis=0).tolist()
        avg_epoch_acc = torch.mean(epoch_acc, axis=0).tolist()
        
        self.logger.info("Validation at Epoch-" + str(self.current_epoch) + " | " + "Average Loss: " + str(
             avg_epoch_loss) + " | " + "Average IoU: " + str(avg_epoch_iou) + ' | ' + 'Number of NaNs: ' + str(
                 cum_epoch_nans) + ' | ' + 'Average Accuracy: ' + str(avg_epoch_acc))

        tqdm_batch.close()
        
        return avg_epoch_iou

    def add_hparams_summary_writer(self):
        '''
        Add Hyperparamters to tensorboard summary writers using .add_hparams
        Can be accessed under the Hyperparameter tab in Tensorboard
        '''

        hyper_params = {
            'loss_func': self.config.loss.func,
            'loss_alpha': self.config.loss.alpha,                                                         
            'loss_gamma': self.config.loss.gamma,    
            'loss_skip_v_every_n_its': self.config.loss.skip_v_every_n_its,
            'loss_skip_p_every_n_its': self.config.loss.skip_p_every_n_its,
            'loss_skip_b_every_n_its': self.config.loss.skip_b_every_n_its,
            'optimizer': self.config.optimizer.func,
            'learning_rate': self.config.optimizer.learning_rate,
            'beta1': self.config.optimizer.beta1,
            'beta2': self.config.optimizer.beta2,
            'eps': self.config.optimizer.eps,
            'amsgrad': self.config.optimizer.amsgrad,
            'weight_decay': self.config.optimizer.weight_decay,
            'lr_scheduler': self.config.optimizer.lr_scheduler.want,
            'lr_scheduler_every_n_epochs': self.config.optimizer.lr_scheduler.every_n_epochs,
            'lr_scheduler_gamma': self.config.optimizer.lr_scheduler.gamma,
        }
       
        self.train_summary_writer.add_hparams(hyper_params, {})
        self.val_summary_writer.add_hparams(hyper_params, {})

    def save_hparams_json(self):
        '''
        Uses config information to generate a hyperparameter dict and saves it as a json file
        into the current_run directory
        '''

        hparams = {
            'loss': self.config.loss,
            'optimizer': self.config.optimizer
        }

        utils.save_json_file(os.path.join(self.config.dir.current_run, 'hyperparams.json'), 
                                hparams , indent=4)
    
    def finalize(self):
        '''
        Close all Writers and print time
        '''
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.train_summary_writer.close()
        self.val_summary_writer.close()
        print('ending ' + self.config.loader.mode + ' at ' + str(datetime.now()))
