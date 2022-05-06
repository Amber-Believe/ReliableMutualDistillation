from torch.utils.data import DataLoader
import torch
from .LIDC import LIDC_loader
from .ACDC import ACDC_loader
from .Brats import Brats_loader
def generate_datasets(args, path):
    batch_size = args.batchSz

    if args.dataset_name == 'LIDC':
        train_loader = LIDC_loader(mode='train', data_path=path)
        val_loader = LIDC_loader(mode='test', data_path=path)

    if args.dataset_name == 'ACDC':
        train_loader = ACDC_loader(mode='train', data_path=path)
        val_loader = ACDC_loader(mode='test', data_path=path)

    if args.dataset_name == 'Brats':
        train_loader = Brats_loader(mode='train', data_path=path)
        val_loader = Brats_loader(mode='test', data_path=path)

    training_generator = torch.utils.data.DataLoader(dataset=train_loader,
                                                          batch_size=batch_size)
    val_generator = torch.utils.data.DataLoader(dataset=val_loader,
                                                batch_size=batch_size)




    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator,  val_generator


