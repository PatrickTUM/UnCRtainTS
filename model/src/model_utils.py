import os
import torch

sub_dir = os.path.join(os.getcwd(), 'model')
if os.path.isdir(sub_dir): os.chdir(sub_dir)
from src.backbones import base_model, utae, uncrtaints

S1_BANDS = 2
S2_BANDS = 13

def get_base_model(config):
    model = base_model.BaseModel(config)
    return model

# for running image reconstruction
def get_generator(config):
    if "unet" in config.model:
            model = utae.UNet(
                input_dim=S1_BANDS*config.use_sar+S2_BANDS,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                out_nonlin_mean=config.mean_nonLinearity,
                out_nonlin_var=config.var_nonLinearity,
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                encoder_norm=config.encoder_norm,
                norm_skip='batch',
                norm_up='batch',
                decoder_norm=config.decoder_norm,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
    elif "utae" in config.model:
        if config.pretrain:
            # on monotemporal data, just use a simple U-Net
            model = utae.UNet(
                input_dim=S1_BANDS*config.use_sar+S2_BANDS, 
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                out_nonlin_mean=config.mean_nonLinearity,
                out_nonlin_var=config.var_nonLinearity,
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                encoder_norm=config.encoder_norm,
                norm_skip='batch',
                norm_up='batch',
                decoder_norm=config.decoder_norm,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        else:
            model = utae.UTAE(
                input_dim=S1_BANDS*config.use_sar+S2_BANDS,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                out_nonlin_mean=config.mean_nonLinearity,
                out_nonlin_var=config.var_nonLinearity,
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                norm_skip='batch',
                norm_up='batch',
                decoder_norm=config.decoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
                positional_encoding=config.positional_encoding,
                scale_by=config.scale_by
            )
    elif 'uncrtaints' == config.model:
        model = uncrtaints.UNCRTAINTS(
                input_dim=S1_BANDS*config.use_sar+S2_BANDS,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths, 
                out_conv=config.out_conv,
                out_nonlin_mean=config.mean_nonLinearity,
                out_nonlin_var=config.var_nonLinearity,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                decoder_norm=config.decoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
                positional_encoding=config.positional_encoding,
                covmode=config.covmode,
                scale_by=config.scale_by,
                separate_out=config.separate_out,
                use_v=config.use_v,
                block_type=config.block_type,
                is_mono=config.pretrain
            )
    else: raise NotImplementedError
    return model


def get_model(config):
    return get_base_model(config)


def save_model(config, epoch, model, name):
    state_dict = {"epoch":          epoch,
                  "state_dict":     model.state_dict(),
                  "state_dict_G":   model.netG.state_dict(),
                  "optimizer_G":    model.optimizer_G.state_dict(),
                  "scheduler_G":    model.scheduler_G.state_dict()}
    torch.save(state_dict,
        os.path.join(config.res_dir, config.experiment_name, f"{name}.pth.tar"),
    )


def load_model(config, model, train_out_layer=True, load_out_partly=True):
    # load pre-trained checkpoints, but only of matching weigths
    
    pretrained_dict = torch.load(config.trained_checkp, map_location=config.device)["state_dict_G"]
    model_dict      = model.netG.state_dict() 

    not_str = "" if pretrained_dict.keys() == model_dict.keys() else "not "
    print(f'The new and the (pre-)trained model architectures are {not_str}identical.\n')

    try:# try loading checkpoint strictly, all weights must match
        # (this is satisfied e.g. when resuming training)

        if train_out_layer: raise NotImplementedError # move to 'except' case
        model.netG.load_state_dict(pretrained_dict, strict=True)
        freeze_layers(model.netG, grad=True)    # set all weights to trainable, no need to freeze
        model.frozen, freeze_these = False, []  # ... as all weights match appropriately
    except: # if some weights don't match (e.g. when loading from pre-trained U-Net), then only load the compatible subset ...
        #     ... freeze compatible weights and make the incompatibel weights trainable

        # load output layer partly, e.g. when pretrained net has 3 output channels but novel model has 13
        if load_out_partly:
            # overwrite output layer even when dimensions mismatch (this overwrites kernels individually)
            #""" # these lines were used for predicting the 13 mean bands when mean and var shared a single output layer
            temp_weights, temp_biases       = model_dict['out_conv.conv.conv.0.weight'], model_dict['out_conv.conv.conv.0.bias']
            temp_weights[:S2_BANDS,...]     = pretrained_dict['out_conv.conv.conv.0.weight'][:S2_BANDS,...]
            temp_biases[:S2_BANDS,...]      = pretrained_dict['out_conv.conv.conv.0.bias'][:S2_BANDS,...]
            pretrained_dict['out_conv.conv.conv.0.weight'] = temp_weights[:S2_BANDS,...]
            pretrained_dict['out_conv.conv.conv.0.bias']   = temp_biases[:S2_BANDS,...]
            """
            if 'out_conv.conv.conv.0.weight' in pretrained_dict: # if predicting from a model with a single output layer for both mean and var
                pretrained_dict['out_conv_mean.conv.conv.0.weight'] = pretrained_dict['out_conv.conv.conv.0.weight'][:S2_BANDS,...]
                pretrained_dict['out_conv_mean.conv.conv.0.bias']   = pretrained_dict['out_conv.conv.conv.0.bias'][:S2_BANDS,...]
            if 'out_conv_var.conv.conv.0.weight' in model_dict:
                pretrained_dict['out_conv_var.conv.conv.0.weight'] = model_dict['out_conv_var.conv.conv.0.weight']
                pretrained_dict['out_conv_var.conv.conv.0.bias']   = model_dict['out_conv_var.conv.conv.0.bias']
            """

        # check for size mismatch and exclude layers whose dimensions mismatch (they won't be loaded)
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict) 
        model.netG.load_state_dict(model_dict, strict=False)
        
        # freeze pretrained weights 
        model.frozen = True
        freeze_layers(model.netG, grad=True) # set all weights to trainable, except final ...
        if train_out_layer:
            # freeze all but last layer
            all_but_last = {k:v for k, v in pretrained_dict.items() if 'out_conv.conv.conv.0' not in k}
            freeze_layers(model.netG, apply_to=all_but_last, grad=False)
            freeze_these = list(all_but_last.keys())
        else: # freeze all pre-trained layers, without exceptions
            freeze_layers(model.netG, apply_to=pretrained_dict, grad=False)
            freeze_these = list(pretrained_dict.keys())
    train_these = [train_layer for train_layer in list(model_dict.keys()) if train_layer not in freeze_these]
    print(f'\nFroze these layers: {freeze_these}')
    print(f'\nTrain these layers: {train_these}')

    if config.resume_from:
        resume_at = int(config.trained_checkp.split('.pth.tar')[0].split('_')[-1])
        print(f'\nResuming training at epoch {resume_at+1}/{config.epochs}, loading optimizers and schedulers')
        # if continuing training, then also load states of previous runs' optimizers and schedulers
        # ---else, we start optimizing from scratch but with the model parameters loaded above
        optimizer_G_dict = torch.load(config.trained_checkp, map_location=config.device)["optimizer_G"]
        model.optimizer_G.load_state_dict(optimizer_G_dict)

        scheduler_G_dict = torch.load(config.trained_checkp, map_location=config.device)["scheduler_G"]
        model.scheduler_G.load_state_dict(scheduler_G_dict)

    # no return value, models are passed by reference


# function to load checkpoints of individual and ensemble models
# (this is used for training and testing scripts)
def load_checkpoint(config, checkp_dir, model, name):
    chckp_path = os.path.join(checkp_dir, config.experiment_name, f"{name}.pth.tar")
    print(f'Loading checkpoint {chckp_path}')
    checkpoint = torch.load(chckp_path, map_location=config.device)["state_dict"]

    try: # try loading checkpoint strictly, all weights & their names must match
        model.load_state_dict(checkpoint, strict=True)
    except:
        # rename keys
        #   in_block1 -> in_block0, out_block1 -> out_block0
        checkpoint_renamed = dict()
        for key, val in checkpoint.items():
            if 'in_block' in key or 'out_block' in key:
                strs    = key.split('.')
                strs[1] = strs[1][:-1] + str(int(strs[1][-1])-1)
                strs[1] = '.'.join([strs[1][:-1], strs[1][-1]])
                key     = '.'.join(strs)
            checkpoint_renamed[key] = val
        model.load_state_dict(checkpoint_renamed, strict=False)

def freeze_layers(net, apply_to=None, grad=False):
    if net is not None:
        for k, v in net.named_parameters():
            # check if layer is supposed to be frozen
            if hasattr(v, 'requires_grad') and v.dtype != torch.int64:
                if apply_to is not None:
                    # flip
                    if k in apply_to.keys() and v.size() == apply_to[k].size(): 
                        v.requires_grad_(grad)
                else: # otherwise apply indiscriminately to all layers
                    v.requires_grad_(grad)