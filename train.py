
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from dataloder import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random
import sys
import pickle
import argparse
import json
import logging
import pandas as pd
from torch import optim
from tqdm import tqdm

from meta import Meta
import matplotlib
matplotlib.use('Agg')


def get_argparser():
    argparser = argparse.ArgumentParser()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int,
                           help='epoch number', default=120000)
    argparser.add_argument('--seed', type=int,
                           help='epoch number', default=222)
    argparser.add_argument('--times', type=int,
                           help='epoch number', default=1)
    argparser.add_argument('--model', type=str,
                           help='model', default='train')
    argparser.add_argument('--sim_type', type=str,
                           help='/GC/random', default='sim_cos')
    argparser.add_argument('--task_type', type=str,
                           help='+/-/+-', default='+')
    argparser.add_argument('--test_epoch', type=int,
                           help='保存的测试点', default=10)
    argparser.add_argument('--test_update', type=int,
                           help='test_update', default=1)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int,
                           help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int,
                           help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int,
                           help='meta batch size, namely task num', default=4)
    argparser.add_argument('--num', type=int,
                           help='meta batch size, namely task num', default=20)
    argparser.add_argument('--meta_lr', type=float,
                           help='meta-level outer learning rate', default=5e-4)
    argparser.add_argument('--update_lr', type=float,
                           help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int,
                           help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int,
                           help='update steps for finetunning', default=5)

    args = argparser.parse_args()

    args = argparser.parse_args()
    # 四个卷积层+一个线性层
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]
    return args, config


def main(db_test, db=None):
    '''
    训练测试保存模型，每训练一个epoch保存一下模型，看看实验结果，画出loss,acc的图
    '''
    args, config = get_argparser()

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    loss_train = []
    loss_test = []
    train_acc = []
    test_acc = []

    for epoch in range(120):
        maml.load_model()
        maml.train()
        '''
        # batchsz here means total episode number

        # train 每次取1000个，姑且算一个epoch
        mini = MiniImagenet('dataset/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz=1000, resize=args.imgsz)

        # fetch meta_batchsz num of episode each time
        print('training')
        db = DataLoader(mini, args.task_num, shuffle=True,
                        num_workers=1, pin_memory=True)

        # 训练
        '''
        tmp_train_loss = 0
        tmp_train_acc = 0

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(
                device), x_qry.to(device), y_qry.to(device)

            acc, loss = maml(x_spt, y_spt, x_qry, y_qry)  # 训练的loss
            tmp_train_loss += loss
            tmp_train_acc += acc

        tmp_train_loss = tmp_train_loss/step
        tmp_train_acc = tmp_train_acc/step

        loss_train.append(tmp_train_loss)
        train_acc.append(tmp_train_acc)

        print(epoch, 'all train loss', tmp_train_loss,
              'all train acc', tmp_train_acc)
        logging.info(
            f'epoch {epoch}:all train loss: {tmp_train_loss},all train acc: {tmp_train_acc}')
        '''
        state = {'net': maml.net.state_dict(
        ), 'optimizer': maml.meta_optim.state_dict(), 'epoch': epoch}
        torch.save(state, f'checkpoint/maml_{args.n_way}_{args.k_spt}_{epoch}.pth.tar')

        # 测试
        '''
        maml.eval()
        print('testing')

        tmp_test_loss = 0
        tmp_test_acc = 0

        # maml.load_model(f'checkpoint/maml_{args.n_way}_{args.k_spt}_{epoch}.pth.tar')

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(
                0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            acc, loss = maml.finetunning(x_spt, y_spt, x_qry, y_qry)  # 测试的loss
            # print(acc,loss)
            # break
            tmp_test_loss += loss
            tmp_test_acc += acc

        tmp_test_loss = tmp_test_loss/step
        tmp_test_acc = tmp_test_acc/step
        print('Test acc:', tmp_test_acc, 'test loss', tmp_test_loss)

        loss_test.append(tmp_test_loss)
        test_acc.append(tmp_test_acc)

        print('Test acc:', tmp_test_acc, 'test loss', tmp_test_loss)
        logging.info(
            f'epoch {epoch}:test acc: {tmp_test_acc},test loss: {tmp_test_loss}')
        break

        # 画图
        x = np.array(range(0, epoch+1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title(f'loss of {epoch}')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()
        #l1 = ax1.plot(x, loss_train, c='r', marker='.')
        l2 = ax1.plot(x, loss_test, c='b', marker='.')
        plt.savefig(f"loss-{args.n_way}-{args.k_spt}.png")

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        plt.title(f'acc of {epoch}')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        plt.grid()
        #l1 = ax1.plot(x, train_acc, c='r', marker='.')
        l2 = ax1.plot(x, test_acc, c='b', marker='.')
        plt.savefig(f"acc-{args.n_way}-{args.k_spt}.png")


def test_single(db_test, db):  # 画图
    '''
    测试单个任务在不同epoch上的性能
    '''

    args, config = get_argparser()

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    for id, (x_test_spt, y_test_spt, x_test_qry, y_test_qry) in enumerate(db_test):

        x_test_spt, y_test_spt, x_test_qry, y_test_qry = x_test_spt.squeeze(0).to(device), y_test_spt.squeeze(
            0).to(device), x_test_qry.squeeze(0).to(device), y_test_qry.squeeze(0).to(device)

        test_loss, test_acc = [], []
        for epoch in range(11):
            #  加载模型
            maml.load_model(
                f'checkpoint/maml_{args.n_way}_{args.k_spt}_{epoch}.pth.tar')
            maml.eval()
            # 得到相应的准确率和loss
            acc, loss = maml.finetunning(
                x_test_spt, y_test_spt, x_test_qry, y_test_qry)
            # print(epoch,loss_full)
            test_loss.append(loss)
            test_acc.append(acc)

        # 二次训练
        maml.train()
        task = pd.read_csv(f"result/sim/task{args.test_epoch}_{args.seed}_{args.sim_type}.csv", index_col=0)
        # train_id=task.iloc[id,0:args.num]
        train_id = list(task.iloc[id, :args.num]) + \
            list(task.iloc[id, -args.num:])
        #train_id=[int(num) for num in train_id]*args.times
        # print(train_id)

        for step in train_id:
            (x_spt, y_spt, x_qry, y_qry) = db.dataset[int(step)]

            x_spt, y_spt, x_qry, y_qry = x_spt.unsqueeze(0).to(device), y_spt.unsqueeze(
                0).to(device), x_qry.unsqueeze(0).to(device), y_qry.unsqueeze(0).to(device)
            maml(x_spt, y_spt, x_qry, y_qry)  # 训练的loss
            # print(acc,loss)

        # 测试
        maml.eval()
        acc, loss = maml.finetunning(
            x_test_spt, y_test_spt, x_test_qry, y_test_qry)
        test_loss.append(loss)
        test_acc.append(acc)

        # 画图

        x = np.array(range(epoch+2))
        fig, ax1 = plt.subplots()
        plt.title(f'loss of {id}')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='tab:red')
        ax1.plot(x, test_loss, c='r', marker='.')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        # we already handled the x-label with ax1
        ax2.set_ylabel('acc', color='blue')
        ax2.plot(x, test_acc, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid()
        plt.savefig(f"result/pic_s/loss-{id}-{args.num}-{args.times}.png")


def baseline(db_test, db):
    '''
    retrain的框架
    '''

    args, config = get_argparser()

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    if args.sim_type!='random':
        task = pd.read_csv(
            f"result/sim/task5000{args.test_epoch}_{args.seed}_{args.sim_type}.csv", index_col=0)
    loss_task = []
    test_loss, test_acc, b_test_loss, b_test_acc = 0, 0, 0, 0
    t_loss, t_acc, b_loss, b_acc = [], [], [], []
    for id, (x_test_spt, y_test_spt, x_test_qry, y_test_qry) in enumerate(db_test):
        maml.load_model(f'checkpoint/maml_5_5_{args.test_epoch}.pth.tar')
        maml.eval()

        # 不经过二次训练的结果
        x_test_spt, y_test_spt, x_test_qry, y_test_qry = x_test_spt.squeeze(0).to(device), y_test_spt.squeeze(
            0).to(device), x_test_qry.squeeze(0).to(device), y_test_qry.squeeze(0).to(device)
        acc, loss = maml.finetunning(
            x_test_spt, y_test_spt, x_test_qry, y_test_qry)
        # print(loss,acc)
        # b_..保存的是不经过二次训练的结果
        b_test_loss += loss
        b_test_acc += acc
        b_loss.append(loss)
        b_acc.append(acc)

        # 选任务二次训练：
        #print(args.task_type)
        if args.sim_type == 'random':
            train_id=np.random.choice(list(range(1000)), args.num, False)
        elif args.task_type == '+-':
            train_id = list(task.iloc[id, :args.num//2]) + \
                list(task.iloc[id, -args.num//2:])
        elif args.task_type == '+':
            train_id = list(task.iloc[id, :args.num])
        elif args.task_type == '-':
            train_id = list(task.iloc[id, -args.num:])
        else:
            raise NotImplementedError
        np.random.shuffle(train_id)

        maml.train()
        for task1 in train_id:
            # print(task1)
            (x_spt, y_spt, x_qry, y_qry) = db.dataset[int(task1)]
            x_spt, y_spt, x_qry, y_qry = x_spt.unsqueeze(0).to(device), y_spt.unsqueeze(
                0).to(device), x_qry.unsqueeze(0).to(device), y_qry.unsqueeze(0).to(device)
            acc_tmp, loss_tmp = maml(x_spt, y_spt, x_qry, y_qry)

        # 测试
        maml.eval()
        acc_test, loss_test = maml.finetunning(
            x_test_spt, y_test_spt, x_test_qry, y_test_qry)
        if loss_test < loss:
            print(id, loss, loss_test, acc, acc_test, '***')
            logging.info(
                f'任务{id}：loss：{loss}，{loss_test}，acc:{acc},{acc_test}****')
        else:
            print(id, loss, loss_test, acc, acc_test)
            logging.info(
                f'任务{id}：loss：{loss}，{loss_test}，acc:{acc},{acc_test}')
        if loss_test > loss:
            loss_task.append(id)

        t_loss.append(loss_test)
        t_acc.append(acc_test)
        test_acc += acc_test
        test_loss += loss_test

    print(len(loss_task), loss_task)
    logging.info(
        f'失败的任务数量:{len(loss_task)}，失败的任务：{loss_task}')
    print(
        f'不经过二次训练所有任务均值 loss:{b_test_loss/(id+1)}，准确率{b_test_acc/(id+1)},方差：{np.var(b_loss)}')
    logging.info(
        f'不经过二次训练所有任务均值 loss:{b_test_loss/(id+1)}，准确率{b_test_acc/(id+1)},方差：{np.var(b_loss)}')
    print(
        f'经过二次训练所有任务均值 loss:{test_loss/(id+1)}，准确率{test_acc/(id+1)},方差：{np.var(t_loss)}')
    logging.info(
        f'经过二次训练所有任务均值 loss:{test_loss/(id+1)}，准确率{test_acc/(id+1)},方差：{np.var(t_loss)}')


def test(db_test, db):

    args, config = get_argparser()

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    test_acc = []
    test_loss = []

    for epoch in range(120):

        # maml.load_model(f'checkpoint/maml_{args.n_way}_{args.k_spt}_{epoch}.pth.tar')
        print('testing')

        tmp_test_loss = 0
        tmp_test_acc = 0

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
            #print(x_spt[0][0][0])
            maml.load_model(f'checkpoint/maml_5_5_{args.test_epoch}.pth.tar')
            maml.eval()
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(
                0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            acc, loss = maml.finetunning(
                x_spt, y_spt, x_qry, y_qry, step)  # 测试的loss
            print(loss, acc)
            tmp_test_loss += loss
            tmp_test_acc += acc

        tmp_test_loss = tmp_test_loss/(step+1)
        tmp_test_acc = tmp_test_acc/(step+1)

        test_loss.append(tmp_test_loss)
        test_acc.append(tmp_test_acc)

        print(epoch, 'Test acc:', tmp_test_acc, 'test loss', tmp_test_loss)
        break


def get_change(maml, x_test_spt, y_test_spt, params):
    args, config = get_argparser()

    # 读取模型
    device = torch.device('cuda')
    params_test = maml.get_test_params_change(x_test_spt, y_test_spt)
    result = torch.cosine_similarity(params, params_test, -1)
    tmp = torch.argsort(result).cpu().numpy()
    task1 = tmp[-1]  # 选择最好的一个任务。
    result1 = torch.cosine_similarity(
        params, params[task1], -1)  # 计算选出来的任务和其他任务的一个距离
    result2 = result-result1
    tmp1 = torch.argsort(result2).cpu().numpy()
    task2 = tmp[-2]
    print(task1, task2)

    return task1, task2


def params_change(db_test, db):

    args, config = get_argparser()

    # 读取模型
    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    maml.load_model(f'checkpoint/maml_5_5_{args.test_epoch}.pth.tar')
    maml.eval()

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    if args.sim_type == 'GC':

        # 直接计算
        params = []
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(
                0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            grad = maml.get_params_change(x_spt, y_spt, x_qry, y_qry)
            params.append(grad)

        params = torch.stack(params).detach()
        # pd.DataFrame(params.cpu().numpy()).to_csv(f"params_qry_{args.test_update}.csv")

    elif args.sim_type == 'sim_ou' or 'sim_cos' or 'sim_dot':
        params = []
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt = x_spt.squeeze(0).to(device)
            params.append(maml.sim_emb(x_spt))

        params = torch.stack(params).detach()
        params = params.view(len(params), args.n_way,
                             args.k_spt, -1)  # (num,n,k,-1)
        # print(params)
        params = params.mean(2)  # (num,n,-1)
        params = params.unsqueeze(1)  # (num,1,n,d)

    print('计算相似度')
    task = np.zeros((len(db_test.dataset), len(params)))
    result = np.zeros((len(db_test.dataset), len(params)))
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
        x_spt, y_spt = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device)
        #x_qry, y_qry = x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        if args.sim_type == 'GC':
            params_test = maml.get_test_params_change(x_spt, y_spt)
            result = torch.cosine_similarity(params, params_test, -1)
        elif args.sim_type == 'sim_cos':
            params_test = maml.sim_emb(x_spt).view(
                args.n_way, args.k_spt, -1).mean(1).unsqueeze(1)  # (n,1,d)
            result1 = torch.cosine_similarity(params, params_test, -1)  # (N,N)
            result1, _ = result1.max(-1)
            result = result1.sum(-1).squeeze(0)
        elif args.sim_type == 'sim_ou':
            params_test = maml.sim_emb(x_spt)
            result = -(torch.pow(params-params_test, 2)).sum(-1)
        elif args.sim_type == 'sim_dot':
            params_test = maml.sim_emb(x_spt)
            result = (params*params_test).sum(-1)

        # print(result)
        tmp = torch.argsort(result).cpu().numpy()  # 从小到大排
        tmp = tmp[::-1]  # 从大到小排
        task[step] = tmp

    pd.DataFrame(task).to_csv(
        f"result/sim/task5000{args.test_epoch}_{args.seed}_{args.sim_type}.csv")
    return task


if __name__ == '__main__':

    # main()
    # test_single()
    logger = logging.getLogger(__name__)
    args, config = get_argparser()
    # Setup logging
    if args.model=='test_change':
        log_name=f"result/log/test-{args.test_epoch}-{args.seed}-{args.sim_type}-{args.num}.log"
        print(log_name)
    else:
        log_name=f"result/log/train.log"
    print(log_name)
    logging.basicConfig(format=None,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=log_name)
    # params_change()
    logging.info('#######################################################################################################')
    logging.info('#######################################################################################################')
    logging.info(f'epoch:{args.test_epoch},seed:{args.seed},sim_type:{args.sim_type},num:{args.num},task_type:{args.task_type},train_task:5000')
    logging.info(args)

    # 固定随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    #固定测试数据并读取
    #test_task=json.load(open('dataset/testtask.json'))
    mini_test = MiniImagenet('dataset/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz,seed=args.seed)
    db_test = DataLoader(mini_test, 1, shuffle=False,
                         num_workers=1, pin_memory=True)

    # 读取训练数据
    mini = MiniImagenet('dataset/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=5000, resize=args.imgsz,seed=args.seed)
    db = DataLoader(mini, 1, shuffle=False,
                    num_workers=1, pin_memory=True)

    if args.model == 'train':
        main(db_test, db)
    elif args.model == 'get_params':
        params_change(db_test, db)
    elif args.model == 'test_change':
        baseline(db_test, db)
    elif args.model == 'test':
        test(db_test, db)
    else:
        test_single(db_test, db)
