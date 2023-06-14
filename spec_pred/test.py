import torch
from tqdm import tqdm
from spec_pred.agent_module import SpecModule
from spec_pred.spec_dataset import DataSource, SpecDataset
from torch.utils.data import Dataset, DataLoader


def main(
    model_path: str, win_len: int = 40, data_path: str = None, device: str = "cpu"
):
    device = torch.device(device)
    datasource = DataSource(
        data_path=data_path,
        split=[0.8, 0.1, 0.1],  # [0.8, 0.1, 0.1]
        spec_range=[73456, 73968],  # (71000,76500)
    )
    test_dataset = SpecDataset(
        train_type="test",
        aug=False,
        datasource=datasource,
        win_len=win_len,
    )
    dataloader = DataLoader(
        test_dataset,
        1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        drop_last=False,
    )
    model = SpecModule.resume_from_checkpoint(model_path, map_location=device)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = batch[0]
        x = x.to(device)
        pred = model.infer(x, win_len, 10)
        print(pred.shape)
        break

if __name__ == "__main__":
    model_path = "/hdd/1/chenc/lid/speech-lid/spec_pred/outputs/2023-03-24/17-10-CNNLSTM/ckpt/last.pt"
    win_len = 40
    data_path = "/hdd/1/chenc/lid/speech-lid/spec_pred/data/data.json"
    device = "cuda:0"
    main(model_path, win_len, data_path, device)
