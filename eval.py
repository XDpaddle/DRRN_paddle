import argparse
import paddle
from models import DRRN
from datasets import EvalDataset
from utils import AverageMeter
from x2paddle.torch2paddle import DataLoader
from utils import denormalize
from utils import PSNR
from utils import load_weights
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--eval-scale', type=int, required=True)
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--U', type=int, default=9)
    parser.add_argument('--num-features', type=int, default=128)
    args = parser.parse_args()

    device = paddle.CUDAPlace(0)
    model = DRRN(B=args.B, U=args.U, num_features=args.num_features).to(device)
    model = load_weights(model, args.weights_file)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    if args.eval_file is not None:
        model.eval()
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with paddle.no_grad():
                preds = model(inputs)
            preds = denormalize(preds.squeeze(0).squeeze(0))
            labels = denormalize(labels.squeeze(0).squeeze(0))
            epoch_psnr.update(PSNR(preds, labels, shave_border=args.
                eval_scale), len(inputs))
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg.numpy()[0]))
