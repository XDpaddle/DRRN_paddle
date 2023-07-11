from x2paddle import torch2paddle
import argparse
import os
import copy
import paddle
from paddle import nn
import paddle.optimizer as optim
from tqdm import tqdm
from models import DRRN
from datasets import TrainDataset
from x2paddle.torch2paddle import DataLoader
from datasets import EvalDataset
from utils import AverageMeter
from utils import denormalize
from utils import PSNR
from utils import load_weights
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--eval-file', type=str)
    parser.add_argument('--eval-scale', type=int)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--U', type=int, default=9)
    parser.add_argument('--num-features', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--clip-grad', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    args.outputs_dir = os.path.join(args.outputs_dir, 'x234')
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    device=paddle.CUDAPlace(0)
    paddle.seed(args.seed)

    model = DRRN(B=args.B, U=args.U, num_features=args.num_features).to(device)
    
    if args.weights_file is not None:
        model = load_weights(model, args.weights_file)
    criterion = paddle.nn.MSELoss() 
    optimizer = paddle.optimizer.SGD(
                                      learning_rate=args.lr,
                                      parameters=model.parameters(), 
                                      weight_decay=0.0001)
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.
        batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
    if args.eval_file is not None:
        eval_dataset = EvalDataset(args.eval_file)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        lr = args.lr * 0.5 ** ((epoch + 1) // 10)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr


        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=len(train_dataset) - len(train_dataset) % args.
            batch_size, ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1)
                )
            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss = criterion(preds, labels) / (2 * len(inputs))
                epoch_losses.update(loss.numpy()[0], len(inputs))
                
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg), lr=lr)
                t.update(len(inputs))

        paddle.save(model.state_dict(), os.path.join(args.outputs_dir,
            'epoch_{}.pdiparams'.format(epoch)))
        
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
            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())
                
    if args.eval_file is not None:
        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr.numpy()[0]))
        paddle.save(best_weights, os.path.join(args.outputs_dir,
            'best.pdiparams'))
