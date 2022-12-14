from dataloaders import dataloaders
import torch

def test_dataloaders():
    for modulename, (datamodule, img_size) in dataloaders.items():
        c, h, w = img_size
        data_manager = datamodule(batch_size=10)
        train_loader = data_manager.train_loader()
        test_loader= data_manager.test_loader()
        
        train_sample, train_labels = next(iter(train_loader))
        assert train_sample.size() == torch.Size((10, c, h, w)), f"{modulename} Train images batch size mismatches"
        assert len(train_labels) == 10, f"{modulename} Train labels batch size mismatches"
        
        test_sample, test_labels = next(iter(train_loader))
        assert test_sample.size() == torch.Size((10, c, h, w)), f"{modulename} Test images batch size mismatches"
        assert len(test_labels) == 10, f"{modulename} Test labels batch size mismatches"
        
        train_loader = data_manager.train_loader()
        test_loader= data_manager.test_loader()
        train_resample, _ = next(iter(train_loader))
        assert not torch.equal(train_sample, train_resample), f"{modulename} Train batch not shuffled"
        test_resample, _ = next(iter(test_loader))
        assert not torch.equal(test_sample, test_resample), f"{modulename} Test batch is shuffled"