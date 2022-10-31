import math
import time

import numpy as np
import torch
import torch.nn as nn
from .other_utils import Logger
from autoattack import checks
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, DiceIndex, MyDiceIndex
import torch.nn.functional as F
from nnunet.utilities.nd_softmax import softmax_helper


def one_hot_old(gt, categories):
    # Check the new function in PyTorch!!!
    size = [*gt.shape] + [categories]
    y = gt.view(-1, 1)
    gt = torch.FloatTensor(y.nelement(), categories).zero_().cuda()
    gt.scatter_(1, y, 1)
    gt = gt.view(size).permute(0, 3, 1, 2).contiguous()
    return gt

def one_hot(src, shape):
    onehot = torch.zeros(shape)
    src = src.long()
    if src.device.type == "cuda":
        onehot = onehot.cuda(src.device.index)
    onehot.scatter_(1, src, 1)
    return onehot

class Dice_metric(nn.Module):
    def __init__(self, eps=1e-5):
        super(Dice_metric, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, logits=True):    
        targets = one_hot(targets, inputs.shape)     
        if logits:
            inputs2 = torch.argmax(softmax_helper(inputs) , dim = 1, keepdim = True)
        inputs2 = one_hot(inputs2, inputs.shape)
        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs2 * targets, dims)
        fps = torch.sum(inputs2 * (1 - targets), dims)
        fns = torch.sum((1 - inputs2) * targets, dims)
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        return loss[:, 1:].mean(dim=1)


class AutoAttack():
    def __init__(self, model, norm_type='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 device='cuda', log_path=None, loss_fn=None):
        self.model = model
        self.norm = norm_type
        assert self.norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)
        self.threshold = 0
        self.dice = Dice_metric()
        self.loss_fn = loss_fn
        self.prev = None

        if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
        
        if not self.is_tf_model:
            from .autopgd import APGDAttack
            self.apgd = APGDAttack(
                self.model, dice_thresh =self.threshold, n_restarts=5, n_iter=100,
                verbose=False, eps=self.epsilon, eot_iter=1,
                rho=.75, seed=self.seed, device=self.device, loss_fn = self.loss_fn, norm_type = self.norm)
            """
            from .fab import FABAttack
            self.fab = FABAttack(
                self.model, dice_thresh=self.threshold, n_target_classes=n_target_classes,
                n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                verbose=False, device=self.device)
            """
            from .square import SquareAttack
            self.square = SquareAttack(
                self.model, dice_thresh=self.threshold, p_init=0.8, n_queries=100,
                eps=self.epsilon, n_restarts=1, seed=self.seed,
                verbose=False, device=self.device, resc_schedule=False,  norm_type = self.norm)
                
    
        else:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=False, logger=self.logger)
            
            from .fab_tf import FABAttack_TF
            self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=False, device=self.device)
        
            from .square import SquareAttack
            self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)
    
        #if version in ['standard', 'plus', 'rand']:
        #    self.set_version(version)
        
    def get_logits_classification(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)
        
    def get_logits(self, x):
        assert not self.is_tf_model
        rawoutput = self.model(x)[0]
        output_softmax = softmax_helper(rawoutput)
        return output_softmax.argmax(1) # in fact this is not the logit
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed
    
    def run_standard_evaluation(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            #y_adv = torch.empty_like(y_orig)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])
                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = [y_orig[i][start_idx:end_idx].clone().to(self.device) for i in range(3)]
                #y = y_orig[start_idx:end_idx].clone().to(self.device) 
                output = self.model(x)
                #y_adv[start_idx: end_idx] = output[0]
                dices = self.dice(output[0], y[0]).to(self.device)
                correct_batch = (dices > self.threshold).to(self.device)
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
            robust_accuracy_dict = {'clean': robust_accuracy}
            
            if self.verbose:
                self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
                    
            x_adv = x_orig.clone().detach()
            startt = time.time()
            # start runing all the attacks   
            self.prev = torch.ones(x_orig.shape[0]).to(self.device)
            
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                n_batches = int(np.ceil(num_robust / bs))
                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)

                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = [y_orig[i][batch_datapoint_idcs].clone().to(self.device) for i in range(3)]
                    #y = y_orig[batch_datapoint_idcs].clone().to(self.device) 

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    
                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'dice'
                        self.apgd.threshold = self.threshold
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True    
                        
                    elif attack == 'apgd-dlr':
                        self.apgd.loss = 'dlr'
                        self.apgd.threshold = self.threshold
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        self.apgd.threshold = self.threshold
                        adv_curr = self.square.perturb(x, y)                                  
                    else:
                        raise ValueError('Attack not supported')
                
                    output = self.model(adv_curr)                   
                    dices = self.dice(output[0], y[0]).to(self.device)
                    false_batch = (dices < self.prev[batch_datapoint_idcs]).to(robust_flags.device)
                    self.prev[batch_datapoint_idcs] = dices
                    #self.threshold = self.prev.min().item()
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    #  only record those samples with lower dice scores than the last ones
                    x_adv[batch_datapoint_idcs[false_batch]] = adv_curr[false_batch].detach().to(x_adv.device)
                    
                    #x_orig[batch_datapoint_idcs] = x_adv[batch_datapoint_idcs]


        return x_adv
        
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()
            
        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))
        
        return acc.item() / x_orig.shape[0]
        
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        
        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            if return_labels:
                adv[c] = (x_adv, y_adv)
            else:
                adv[c] = x_adv
            if verbose_indiv:    
                acc_indiv  = self.clean_accuracy(x_adv, y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))
        
        return adv
        
    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))
        
        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            #self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
        
        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))
        
        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20

