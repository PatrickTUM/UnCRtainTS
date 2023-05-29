import torch
import torch.nn as nn

from src import losses, model_utils
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

S2_BANDS = 13

class BaseModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super(BaseModel, self).__init__()
        self.config     = config    # store config
        self.frozen     = False     # no parameters are frozen
        self.len_epoch  = 0         # steps of one epoch

        # temporarily rescale model inputs & outputs by constant factor, e.g. from [0,1] to [0,100],
        #       to deal with numerical imprecision issues closeby 0 magnitude (and their inverses)
        #       --- convert inputs, mean & variance predictions to original scale again after NLL loss is computed
        # note: this may also require adjusting the range of output nonlinearities in the generator network,
        #       i.e. out_mean, out_var and diag_var

        #   -------------- set input via set_input and call forward ---------------
        # inputs self.real_A & self.real_B  set in set_input            by * self.scale_by
        #   ------------------------------ then scale -----------------------------
        # output self.fake_B will automatically get scaled              by ''
        #   ------------------- then compute loss via get_loss_G ------------------
        # output self.netG.variance  will automatically get scaled      by * self.scale_by**2
        #   ----------------------------- then rescale ----------------------------
        # inputs self.real_A & self.real_B  set in set_input            by * 1/self.scale_by
        # output self.fake_B                set in self.forward         by * 1/self.scale_by
        # output self.netG.variance         set in get_loss_G           by * 1/self.scale_by**2
        self.scale_by  = config.scale_by                    # temporarily rescale model inputs by constant factor, e.g. from [0,1] to [0,100]

        # fetch generator
        self.netG = model_utils.get_generator(self.config)

        # 1 criterion
        self.criterion = losses.get_loss(self.config)
        self.log_vars  = None

        # 2 optimizer: for G
        paramsG = [{'params': self.netG.parameters()}]

        self.optimizer_G = torch.optim.Adam(paramsG, lr=config.lr)

        # 2 scheduler: for G, note: stepping takes place at the end of epoch
        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma=self.config.gamma)

        self.real_A = None
        self.fake_B = None
        self.real_B = None
        self.dates  = None
        self.masks  = None
        self.netG.variance = None

    def forward(self):
        # forward through generator, note: for val/test splits, 
        # 'with torch.no_grad():' is declared in train script
        self.fake_B = self.netG(self.real_A, batch_positions=self.dates)
        if self.config.profile: 
            flopstats  = FlopCountAnalysis(self.netG, (self.real_A, self.dates))
            # print(flop_count_table(flopstats))
            # TFLOPS: flopstats.total() *1e-12
            # MFLOPS: flopstats.total() *1e-6
            # compute MFLOPS per input sample
            self.flops = (flopstats.total()*1e-6)/self.config.batch_size
            print(f"MFLOP count: {self.flops}")
        self.netG.variance = None # purge earlier variance prediction, re-compute via get_loss_G()

    def backward_G(self):
        # calculate generator loss
        self.get_loss_G()
        self.loss_G.backward()


    def get_loss_G(self):

        if hasattr(self.netG, 'vars_idx'):
            self.loss_G, self.netG.variance = losses.calc_loss(self.criterion, self.config, self.fake_B[:, :, :self.netG.mean_idx, ...], self.real_B, var=self.fake_B[:, :, self.netG.mean_idx:self.netG.vars_idx, ...])
        else: # used with all other models
            self.loss_G, self.netG.variance = losses.calc_loss(self.criterion, self.config, self.fake_B[:, :, :S2_BANDS, ...], self.real_B, var=self.fake_B[:, :, S2_BANDS:, ...])

    def set_input(self, input):
        self.real_A = self.scale_by * input['A'].to(self.config.device)
        self.real_B = self.scale_by * input['B'].to(self.config.device)
        self.dates  = None if input['dates'] is None else input['dates'].to(self.config.device)
        self.masks  = input['masks'].to(self.config.device)


    def reset_input(self):
        self.real_A = None
        self.real_B = None
        self.dates  = None 
        self.masks  = None
        del self.real_A
        del self.real_B 
        del self.dates
        del self.masks


    def rescale(self):
        # rescale target and mean predictions
        if hasattr(self, 'real_A'): self.real_A = 1/self.scale_by * self.real_A
        self.real_B = 1/self.scale_by * self.real_B 
        self.fake_B = 1/self.scale_by * self.fake_B[:,:,:S2_BANDS,...]
        
        # rescale (co)variances
        if hasattr(self.netG, 'variance') and self.netG.variance is not None:
            self.netG.variance = 1/self.scale_by**2 * self.netG.variance

    def optimize_parameters(self):
        self.forward()
        del self.real_A

        # update G
        self.optimizer_G.zero_grad() 
        self.backward_G()
        self.optimizer_G.step()

        # re-scale inputs, predicted means, predicted variances, etc
        self.rescale()
        # resetting inputs after optimization saves memory
        self.reset_input()

        if self.netG.training: 
            self.fake_B = self.fake_B.cpu()
            if self.netG.variance is not None: self.netG.variance = self.netG.variance.cpu()