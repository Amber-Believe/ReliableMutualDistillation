import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch.optim as optim

import lib.medloaders as medical_loaders
# Lib files
import lib.utils as utils
from lib.losses3D import BCEDiceLoss
from lib.train.p_train_our import Trainer
from lib.medzoo.Unet2D import Unet
from lib.medzoo.dense_unet_model import Dense_Unet
from lib.medzoo.myresnetunetplus import ResNetUnetPlus
import torch
import numpy as np

#This one for training. Amber

def main():
    args = get_arguments()

    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)

    utils.make_dirs(args.save)
    utils.save_arguments(args, args.save)
    training_generator_weak, val_generator,_ = medical_loaders.generate_datasets(args,path='/home/qwang/Desktop/code/datasets/LIDC_data/noisy_0.6')


    # unet
    model1 = Unet(1, 1)
    model2 = Unet(1, 1)
    #Dense_Unet
    #model1 = Dense_Unet(in_chan=1, out_chan=1, filters=64, num_conv=4)
    #model2 = Dense_Unet(in_chan=1, out_chan=1, filters=64, num_conv=4)

    #ResNetUnetPlus
    #model1 = ResNetUnetPlus(num_channels=1, out_ch=1)
    #model2 = ResNetUnetPlus(num_channels=1, out_ch=1)
    #optimizer1 = optim.Adam(list(model1.parameters()) + list(model2.parameters()), 0.001)
    #optimizer1 = optim.SGD(list(model1.parameters()) + list(model2.parameters()), 1e-3, momentum=0.9, weight_decay=0.0001)
    #optimizer1 = optim.SGD(list(model1.parameters()) + list(model2.parameters()), 0.3, momentum=0.9,weight_decay=0.0001)
    optimizer1 = optim.Adam(model1.parameters(), 1e-3)
    optimizer2 = optim.Adam(model2.parameters(), 1e-3)

    if args.forget_rate is None:
        args.forget_rate = args.noise_rate
    else:
        args.forget_rate = args.forget_rate
    # define drop rate schedule
    rate_schedule = np.ones(args.nEpochs) * args.forget_rate
    rate_schedule[:args.num_gradual] = np.linspace(0, args.forget_rate ** args.exponent, args.num_gradual)
    '''choose DiceLoss or BCEDiceLoss'''
    criterion = BCEDiceLoss(classes=args.classes)
    if args.cuda:
        model1 = model1.cuda()
        model2 = model2.cuda()
        #vnet1 = vnet1.cuda)
        #vnet2 = vnet2.cuda()
        print("Model transferred in GPU.....")
    trainer = Trainer(args, model1, model2,rate_schedule,criterion, optimizer1,optimizer2,
                      train_data_loader_weak=training_generator_weak,
                        valid_data_loader=val_generator ,lr_scheduler=None)
    #trainer = Trainer(args, model1, criterion, optimizer1,train_data_loader=training_generator_weak,
    #                        valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_aug', type=int, default=1)
    parser.add_argument('--consist_factor', type=float, default=1)
    parser.add_argument('--consistency_rampup', type=int, default=10)
    parser.add_argument('--consistency', type=float, default=1)
    parser.add_argument('--batchSz', type=int, default=20)
    parser.add_argument('--dataset_name', type=str, default="LIDC")
    parser.add_argument('--nEpochs', type=int, default=50)
    parser.add_argument('--epoch_decay_start', type=int, default=10)

    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--inChannels', type=int, default=1)  #default=4
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate (default: 1e-3)')#default=5e-3
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET2D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET')) #
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop')) #
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.005)
    parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
    parser.add_argument('--num_gradual', type=int, default=10,
                        help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type=float, default=1,
                        help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    args.tb_log_dir = '../runs/'
    return args


if __name__ == '__main__':

    main()