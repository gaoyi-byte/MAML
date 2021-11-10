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

import matplotlib.image as mpimg

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def get_argparser():
    argparser = argparse.ArgumentParser()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int,
                           help='epoch number', default=120000)
    argparser.add_argument('--times', type=int,
                           help='epoch number', default=1)
    argparser.add_argument('--model', type=str,
                           help='model', default='train')
    argparser.add_argument('--sim_type', type=str,
                           help='sim_type', default='sim_cos')
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
                           help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float,
                           help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float,
                           help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int,
                           help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int,
                           help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    args = argparser.parse_args()
    #四个卷积层+一个线性层
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

def main():

    # 固定随机种子
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    random.seed(222)
    torch.backends.cudnn.deterministic = True

    args, config = get_argparser()

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    mini_test = MiniImagenet('dataset/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    db_test = DataLoader(mini_test, 1, shuffle=True,
                             num_workers=1, pin_memory=True)
    loss_train = []
    loss_test = []
    train_acc = []
    test_acc = []

    for epoch in range(120):
        # batchsz here means total episode number
        '''

        # train 每次取1000个，姑且算一个epoch
        mini = MiniImagenet('dataset/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz=1000, resize=args.imgsz)

        # fetch meta_batchsz num of episode each time
        print('training')
        db = DataLoader(mini, args.task_num, shuffle=True,
                        num_workers=1, pin_memory=True)

        # 训练
        tmp_train_loss = 0
        tmp_train_acc = 0

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(
                device), x_qry.to(device), y_qry.to(device)

            acc, loss = maml(x_spt, y_spt, x_qry, y_qry)  # 训练的loss
            print(step,acc,loss)
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

        state = {'net': maml.net.state_dict(
        ), 'optimizer': maml.meta_optim.state_dict(), 'epoch': epoch}
        torch.save(state, f'checkpoint/maml_{args.n_way}_{args.k_spt}_{epoch}.pth.tar')

        # 测试
        
        '''
        print('testing')

        tmp_test_loss = 0
        tmp_test_acc = 0
        
        maml.load_model()

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(
                0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            acc, loss = maml.finetunning(x_spt, y_spt, x_qry, y_qry)  # 测试的loss
            #print(acc,loss)
            #break
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

        # 画图
        x = np.array(range(0, epoch+1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title(f'loss of {epoch}')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()
        l1 = ax1.plot(x, loss_train, c='r', marker='.')
        l2 = ax1.plot(x, loss_test, c='b', marker='.')
        plt.savefig(f"loss-{args.n_way}-{args.k_spt}.png")

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        plt.title(f'acc of {epoch}')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        plt.grid()
        l1 = ax1.plot(x, train_acc, c='r', marker='.')
        l2 = ax1.plot(x, test_acc, c='b', marker='.')
        plt.savefig(f"acc-{args.n_way}-{args.k_spt}.png")

def train_epoch():

    # 固定随机种子
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    random.seed(222)
    torch.backends.cudnn.deterministic = True

    args, config = get_argparser()

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    mini_test = MiniImagenet('dataset/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    db_test = DataLoader(mini_test, 1, shuffle=False,
                             num_workers=1, pin_memory=True)

    test_task=pd.read_csv('dataset/test_task.csv',index_col=0)

    mini2 = MiniImagenet('dataset/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz=args.num, resize=args.imgsz,task=test_task,num=args.num)
    db2 = DataLoader(mini2, 1, shuffle=False,
                        num_workers=1, pin_memory=True)

    

    for epoch in range(1):
        maml.load_model()

        loss1,acc1=[],[]
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(
                0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            acc, loss = maml.finetunning(x_spt, y_spt, x_qry, y_qry)  # 测试的loss
            loss1.append(loss)
            acc1.append(acc)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db2):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(
                device), x_qry.to(device), y_qry.to(device)

            acc, loss = maml(x_spt, y_spt, x_qry, y_qry)  # 训练的loss

        # 测试
        print('testing')
        loss2,acc2=[],[]
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(
                0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            acc, loss = maml.finetunning(x_spt, y_spt, x_qry, y_qry)  # 测试的loss
            loss2.append(loss)
            acc2.append(acc)
            
    task=[]
    for i in range(len(loss1)):
        if loss1[i]<loss2[i]:
            task.append(i)
        print(i,loss1[i],loss2[i],acc1[i],acc2[i])
        logging.info(f"{i},{loss1[i]},{loss2[i]},{acc1[i]},{acc2[i]}")
    print(np.mean(loss1),np.mean(loss2),np.mean(acc1),np.mean(acc2))
    logging.info(f"{np.mean(loss1)},{np.mean(loss2)},{np.mean(acc1)},{np.mean(acc2)}")
    print(len(task),task)
    logging.info(f"{len(task)},{task}")

def test_single():#画图
    # 固定随机种子
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    random.seed(222)
    torch.backends.cudnn.deterministic = True

    args, config = get_argparser()

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    #读取测试数据
    mini_test = MiniImagenet('dataset/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    db_test = DataLoader(mini_test, 1, shuffle=True,
                             num_workers=1, pin_memory=True)

    #读取训练数据
    mini = MiniImagenet('dataset/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz=1000, resize=args.imgsz)
    db = DataLoader(mini, 1, shuffle=True,
                        num_workers=1, pin_memory=True)

    for id, (x_test_spt, y_test_spt, x_test_qry, y_test_qry) in enumerate(db_test):

        x_test_spt, y_test_spt, x_test_qry, y_test_qry = x_test_spt.squeeze(0).to(device), y_test_spt.squeeze(
        0).to(device), x_test_qry.squeeze(0).to(device), y_test_qry.squeeze(0).to(device)

        test_loss,test_acc=[],[]
        for epoch in range(120):
            #  加载模型 
            maml.load_model(f'checkpoint/maml_{args.n_way}_{args.k_spt}_{epoch}.pth.tar')
            # 得到相应的准确率和loss
            acc,loss= maml.finetunning(x_test_spt, y_test_spt, x_test_qry, y_test_qry)
            #print(epoch,loss_full)
            test_loss.append(loss)
            test_acc.append(acc)
        
        #二次训练
        task=pd.read_csv(f"test-task{args.test_update}.csv",index_col=0)
        train_id=task.iloc[id,0:args.num]
        train_id=[int(num) for num in train_id]*args.times
        #print(train_id)
        
        for step in train_id:
            (x_spt, y_spt, x_qry, y_qry)=db.dataset[int(step)]
            
            x_spt, y_spt, x_qry, y_qry = x_spt.unsqueeze(0).to(device), y_spt.unsqueeze(
            0).to(device), x_qry.unsqueeze(0).to(device), y_qry.unsqueeze(0).to(device)
            maml(x_spt, y_spt, x_qry, y_qry)  # 训练的loss
            #print(acc,loss)

        #测试
        acc,loss= maml.finetunning(x_test_spt, y_test_spt, x_test_qry, y_test_qry)
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

        ax2.set_ylabel('acc', color='blue')  # we already handled the x-label with ax1
        ax2.plot(x, test_acc, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid()
        plt.savefig(f"outdirpic/test/loss-{id}-{args.num}-{args.times}.png")

def baseline():

    #固定随机种子
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    random.seed(222)
    torch.backends.cudnn.deterministic = True 

    args, config = get_argparser()


    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    

    #读取测试数据
    mini_test = MiniImagenet('dataset/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    db_test = DataLoader(mini_test, 1, shuffle=False,
                             num_workers=1, pin_memory=True)

    #读取训练数据
    mini = MiniImagenet('dataset/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz=1000, resize=args.imgsz) 
    db = DataLoader(mini, 1, shuffle=False,
                        num_workers=1, pin_memory=True)

    '''
    test_task=pd.read_csv('dataset/test_task.csv',index_col=0)
    mini2 = MiniImagenet('dataset/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz=100, resize=args.imgsz,task=test_task,num=args.num)
    db2 = DataLoader(mini2, 1, shuffle=False,
                        num_workers=1, pin_memory=True)

    params=pd.read_csv(f"params_qry_{args.test_update}.csv",index_col=0)
    params = np.array(params)
    params = torch.tensor(params)
    params=params.cuda()
    print(params.shape)
    print(f"读取params_qry_{args.test_update}.csv成功")
    '''


    task=pd.read_csv(f"task_{args.sim_type}.csv",index_col=0)
    loss_task=[]
    test_loss,test_acc,b_test_loss,b_test_acc=0,0,0,0
    t_loss,t_acc,b_loss,b_acc=[],[],[],[]
    step=0
    for id, (x_test_spt, y_test_spt, x_test_qry, y_test_qry) in enumerate(db_test):
        maml.load_model()
        maml.eval()
        
        #不经过二次训练的结果
        x_test_spt, y_test_spt, x_test_qry, y_test_qry = x_test_spt.squeeze(0).to(device), y_test_spt.squeeze(
        0).to(device), x_test_qry.squeeze(0).to(device), y_test_qry.squeeze(0).to(device)
        acc,loss= maml.finetunning(x_test_spt, y_test_spt, x_test_qry, y_test_qry)
        #print(loss,acc)
        #b_..保存的是不经过二次训练的结果
        b_test_loss+=loss
        b_test_acc+=acc
        b_loss.append(loss)
        b_acc.append(acc)
        #print(f'不经过二次训练的任务{id}的loss为{loss}，准确性为{acc}')
        #logging.info(f'不经过二次训练的任务{id}的loss为{loss}，准确性为{acc}')
        
         

        #二次训练
        '''
        #task1,task2=get_change(maml,x_test_spt, y_test_spt,params)
        task1,task2=task.iloc[id,0:2]
        print(task1,task2)

       
        (x_spt, y_spt, x_qry, y_qry)=db.dataset[int(task1)]
        x_spt, y_spt, x_qry, y_qry = x_spt.unsqueeze(0).to(device), y_spt.unsqueeze(0).to(device), x_qry.unsqueeze(0).to(device), y_qry.unsqueeze(0).to(device)
        acc_tmp, loss_tmp = maml(x_spt, y_spt, x_qry, y_qry)  # 训练的loss
        (x_spt, y_spt, x_qry, y_qry)=db.dataset[int(task2)]
        x_spt, y_spt, x_qry, y_qry = x_spt.unsqueeze(0).to(device), y_spt.unsqueeze(0).to(device), x_qry.unsqueeze(0).to(device), y_qry.unsqueeze(0).to(device)
        acc_tmp, loss_tmp = maml(x_spt, y_spt, x_qry, y_qry)  # 训练的loss
        '''
        
        #选任务二次训练：
        #train_id=task.iloc[id,:args.num]
        train_id=np.random.choice(list(range(1000)), args.num, False)
        for task1 in train_id:
            #print(task1)
            (x_spt, y_spt, x_qry, y_qry)=db.dataset[int(task1)]
            x_spt, y_spt, x_qry, y_qry = x_spt.unsqueeze(0).to(device), y_spt.unsqueeze(0).to(device), x_qry.unsqueeze(0).to(device), y_qry.unsqueeze(0).to(device)
            acc_tmp, loss_tmp = maml(x_spt, y_spt, x_qry, y_qry)
        '''
        for _ in range(args.num):
            (x_spt, y_spt, x_qry, y_qry)=db2.dataset[step]
            x_spt, y_spt, x_qry, y_qry = x_spt.unsqueeze(0).to(device), y_spt.unsqueeze(0).to(device), x_qry.unsqueeze(0).to(device), y_qry.unsqueeze(0).to(device)
            acc_tmp, loss_tmp = maml(x_spt, y_spt, x_qry, y_qry)
            step+=1
            print(step)  
        '''

        #测试
        acc_test,loss_test= maml.finetunning(x_test_spt, y_test_spt, x_test_qry, y_test_qry)
        if loss_test<loss:
            print(id,loss,loss_test,acc,acc_test,'***')
            logging.info(f'任务{id}：loss：{loss}，{loss_test}，acc:{acc},{acc_test}****')
        else:
            print(id,loss,loss_test,acc,acc_test)
            logging.info(f'任务{id}：loss：{loss}，{loss_test}，acc:{acc},{acc_test}')
        if loss_test>loss:
            loss_task.append(id)
        


        #print(f'经过二次训练的任务{id}的loss为{loss_test}，准确性为{acc}')
        #logging.info(f'经过二次训练的任务{id}的loss为{loss_test}，准确性为{acc}')
        t_loss.append(loss_test)
        t_acc.append(acc_test)
        test_acc+=acc_test
        test_loss+=loss_test
        
    print(len(loss_task),loss_task)
    logging.info(f'失败的任务数量：{len(loss_task)},任务：{loss_task}')
    x = np.array(range(id+1))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('task')
    plt.ylabel('loss')
    plt.grid()
    l2 = ax1.plot(x, b_loss, c='b', marker='.',linestyle='--',label='before')
    l1 = ax1.plot(x, t_loss, c='r', marker='.',label='after')
    plt.legend()
    plt.savefig(f"loss-{args.test_update}-{args.num}-{args.times}.png")


    print(f'不经过二次训练所有任务均值 loss:{b_test_loss/(id+1)}，准确率{b_test_acc/(id+1)},方差：{np.var(b_loss)}')
    logging.info(f'不经过二次训练所有任务均值 loss:{b_test_loss/(id+1)}，准确率{b_test_acc/(id+1)},方差：{np.var(b_loss)}')
    print(f'经过二次训练所有任务均值 loss:{test_loss/(id+1)}，准确率{test_acc/(id+1)},方差：{np.var(t_loss)}')
    logging.info(f'经过二次训练所有任务均值 loss:{test_loss/(id+1)}，准确率{test_acc/(id+1)},方差：{np.var(t_loss)}')
    

def test():

    #固定随机种子
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    random.seed(222)
    torch.backends.cudnn.deterministic = True

    args, config = get_argparser()


    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    

    #读取测试数据
    mini_test = MiniImagenet('dataset/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    db_test = DataLoader(mini_test, 1, shuffle=False,
                             num_workers=1, pin_memory=True)

    #读取训练数据
    mini = MiniImagenet('dataset/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz=1000, resize=args.imgsz)
    db = DataLoader(mini, 1, shuffle=False,
                        num_workers=1, pin_memory=True)
    
    test_acc = []
    test_loss=[]

    for epoch in range(120):

        #maml.load_model(f'checkpoint/maml_{args.n_way}_{args.k_spt}_{epoch}.pth.tar')
        print('testing')
        maml.load_model()
        maml.eval()

        tmp_test_loss = 0
        tmp_test_acc = 0

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(
                0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            acc, loss = maml.finetunning(x_spt, y_spt, x_qry, y_qry)  # 测试的loss
            print(loss,acc)
            tmp_test_loss += loss
            tmp_test_acc += acc

        tmp_test_loss = tmp_test_loss/step
        tmp_test_acc = tmp_test_acc/step

        test_loss.append(tmp_test_loss)
        test_acc.append(tmp_test_acc)

        print(epoch,'Test acc:', tmp_test_acc, 'test loss', tmp_test_loss)
        break
    '''
    x = np.array(range(epoch+1))
    fig, ax1 = plt.subplots()
    plt.title(f'loss of {id}')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color='tab:red')
    ax1.plot(x, test_loss, c='r', marker='.')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('acc', color='blue')  # we already handled the x-label with ax1
    ax2.plot(x, test_acc, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid()
    plt.savefig(f"outdirpic/result-5-5.png")
    '''
        
def get_change(maml,x_test_spt, y_test_spt,params):
    args, config = get_argparser()

    #读取模型
    device = torch.device('cuda')
    params_test=maml.get_test_params_change(x_test_spt, y_test_spt)
    result=torch.cosine_similarity(params,params_test,-1)
    tmp=torch.argsort(result).cpu().numpy()
    task1=tmp[-1]#选择最好的一个任务。
    result1=torch.cosine_similarity(params,params[task1],-1)#计算选出来的任务和其他任务的一个距离
    result2=result-result1
    tmp1=torch.argsort(result2).cpu().numpy()
    task2=tmp[-2]
    print(task1,task2)
        
    return task1,task2

def params_change():
    # 固定随机种子
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    random.seed(222)
    torch.backends.cudnn.deterministic = True

    args, config = get_argparser()

    #读取模型
    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    maml.load_model()
    maml.eval()

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    #读取测试数据
    mini_test = MiniImagenet('dataset/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    db_test = DataLoader(mini_test, 1, shuffle=False,
                             num_workers=1, pin_memory=True)
    #读取训练数据
    mini = MiniImagenet('dataset/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=1000, resize=args.imgsz)
    db = DataLoader(mini, 1, shuffle=False,
                             num_workers=1, pin_memory=True) 
    #db_test=db
    print('读取完训练数据和测试数据,计算embedding')
    
    
    if args.sim_type=='GC':
        
        #直接计算
        params=[]
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device),x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            grad=maml.get_params_change(x_spt, y_spt, x_qry, y_qry)
            params.append(grad)

        params=torch.stack(params).detach()
        #pd.DataFrame(params.cpu().numpy()).to_csv(f"params_qry_{args.test_update}.csv")
        '''
        #计算保存下来
        params=pd.read_csv(f"params{args.test_update}.csv",index_col=0)
        params = np.array(params)
        params = torch.tensor(params)
        params=params.cuda()
        print(params.shape)
        print(f"读取params{args.test_update}.csv成功")
        '''
    elif args.sim_type=='sim_ou'or'sim_cos'or'sim_dot':
        params=[]
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt= x_spt.squeeze(0).to(device)
            params.append(maml.sim_emb(x_spt))
            #if step==0:
                #a1=maml.sim_emb(x_spt)

        params=torch.stack(params).detach()
        params=params.view(len(params),args.n_way,args.k_spt,-1)#(num,n,k,-1)
        #print(params)
        params=params.mean(2)#(num,n,-1)
        #print(params)
   
        #print(-(torch.pow(a1-params, 2)).sum(-1),-(torch.pow(a1-params[0], 2)).sum(-1))
        #print(torch.cosine_similarity(a1,params,-1),torch.cosine_similarity(a1,params[0],-1))
    
        params=params.unsqueeze(1)#(num,1,n,d)
    
    print('计算相似度')
    task=np.zeros((len(db_test.dataset),len(params)))
    result=np.zeros((len(db_test.dataset),len(params)))
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
        x_spt, y_spt = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device)    
        #x_qry, y_qry = x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        if args.sim_type=='GC':
            params_test=maml.get_test_params_change(x_spt, y_spt)
            result=torch.cosine_similarity(params,params_test,-1)
        elif args.sim_type=='sim_cos':
            params_test=maml.sim_emb(x_spt).view(args.n_way,args.k_spt,-1).mean(1).unsqueeze(1)#(n,1,d)
            #print(params.shape,params_test.shape)
            result1=torch.cosine_similarity(params,params_test,-1)#(N,N)
            result1,_=result1.max(-1)
            result=result1.sum(-1).squeeze(0)
            #print(step,result.shape,result)
            
                
        elif args.sim_type=='sim_ou':
            params_test=maml.sim_emb(x_spt)
            result=-(torch.pow(params-params_test, 2)).sum(-1)
        elif args.sim_type=='sim_dot':
            params_test=maml.sim_emb(x_spt)
            result=(params*params_test).sum(-1)
        
        #print(result)
        tmp=torch.argsort(result).cpu().numpy()#从小到大排
        tmp=tmp[::-1]#从大到小排
        task[step]=tmp

    pd.DataFrame(task).to_csv(f"task_{args.sim_type}.csv")
    return task


if __name__ == '__main__':

    #main()
    #test_single()
    logger = logging.getLogger(__name__)
    args, config = get_argparser()
    # Setup logging
    logging.basicConfig(format=None,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=f"result/test-{args.model}-{args.test_update}-{args.num}_1.log")
    #params_change()
    logging.info(args)
    
    

    if args.model=='train':
        main()
    elif args.model=='get_params':
        params_change() 
    elif args.model=='test_change':
        baseline()  
    elif args.model=='test':
        test()
    else:
        train_epoch()

