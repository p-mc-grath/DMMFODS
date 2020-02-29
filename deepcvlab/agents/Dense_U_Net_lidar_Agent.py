import os
import logging
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

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
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none').cuda()
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.config.optimizer.learning_rate, 
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2), 
            eps=self.config.optimizer.eps, weight_decay=self.config.optimizer.weight_decay.value, 
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
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.dir.pretrained_weights))
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
            loss_per_class = torch.sum(current_loss, dim=(0,2,3))
            epoch_loss += [epoch_loss[-1] + loss_per_class]

            # whole image IoU per class
            iou_per_class = utils.compute_IoU_whole_img_batch(prediction, ht_map, self.config.agent.iou_threshold)
            epoch_iou += [epoch_iou[-1] + iou_per_class]

            # backprop
            self.optimizer.zero_grad()
            current_loss.backward(torch.ones_like(current_loss, device=self.device))                            # , retain_graph=True?
            self.optimizer.step()

            # counters
            self.current_iteration += 1
            current_batch += 1

            # log
            info_per_class_dict = {
                'loss vehicle': loss_per_class[0],
                'loss pedestrian': loss_per_class[1],
                'loss cyclist': loss_per_class[2],
                'iou vehicle': iou_per_class[0],
                'iou pedestrian': iou_per_class[1],
                'iou cyclist': iou_per_class[2]
            }
            self.summary_writer.add_scalars("Training_Info/", info_per_class_dict, self.current_iteration)

        tqdm_batch.close()

        # learning rate decay update; after validate; after each epoch
        self.lr_scheduler.step()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "average loss: " + str(
             epoch_loss[-1]/len(epoch_loss)) + " | " + "average IoU: " + str(epoch_iou[-1]/len(epoch_iou)))

    def validate(self):
        """
        One epoch validation
        :return: 
            average IoU per class
        """
        # Initialize progress visualization and get batch
        # !self.data_loader.valid_loader works for both valid and test 
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = [torch.zeros(self.config.model.num_classes).to(self.device)]
        epoch_iou = [torch.zeros(self.config.model.num_classes)]
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
            loss_per_class = torch.sum(current_loss, dim=(0,2,3))
            epoch_loss += [epoch_loss[-1] + loss_per_class]
            
            # whole image IoU per class
            iou_per_class = utils.compute_IoU_whole_img_batch(prediction, ht_map, self.config.agent.iou_threshold)
            epoch_iou += [epoch_iou[-1] + iou_per_class]
 
        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "average loss: " + str(
             epoch_loss[-1]/len(epoch_loss)) + " | " + "average IoU: " + str(epoch_iou[-1]/len(epoch_iou)))

        tqdm_batch.close()
        
        return epoch_iou[-1]/len(epoch_iou)

    def finalize(self):
        """
        Save checkpoint and log
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.close()
