import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:3',help='')
parser.add_argument('--data',type=str,default='data/BJ-Flow',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=1024,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='modelSave/BJ-Flow.pth',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--onlyEMC', type=bool, default=False, help='只使用EC矩阵')
parser.add_argument('--onlyNMC', type=bool, default=False, help='只使用NMC矩阵')
parser.add_argument('--onlyADP', type=bool, default=False, help='只使用adp矩阵')
parser.add_argument('--hops', type=int, default=2, help='GCN hops')
parser.add_argument('--Conv_bool', type=bool, default=True, help="Converge Time ")
parser.add_argument('--ConvTxt', type=str, default='ConvTxt/BJ-Conv.txt', help="store the converge time")

args = parser.parse_args()




def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = None

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None



    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, onlyEMC=args.onlyEMC, onlyNMC=args.onlyNMC, onlyAdp=args.onlyADP, hops=args.hops)
    best_valid_loss = 10000000


    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    if args.Conv_bool:
        f = open(args.ConvTxt, "w")
        iterNum = 1
        for i in range(1, args.epochs + 1):
            # if i % 10 == 0:
            # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            # for g in engine.optimizer.param_groups:
            # g['lr'] = lr
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)
                trainx = trainx.transpose(1, 3)  # batch*2*T*N
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)
                metrics = engine.train(trainx, trainy[:, 0, :, :])
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                if iter % args.print_every == 0:
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

                if iterNum % args.print_every == 0:
                    valid_loss = []
                    valid_mape = []
                    valid_rmse = []

                    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                        testx = torch.Tensor(x).to(device)
                        testx = testx.transpose(1, 3)
                        testy = torch.Tensor(y).to(device)
                        testy = testy.transpose(1, 3)
                        metrics = engine.eval(testx, testy[:, 0, :, :])
                        valid_loss.append(metrics[0])
                        valid_mape.append(metrics[1])
                        valid_rmse.append(metrics[2])
                    mvalid_loss = np.mean(valid_loss)
                    mvalid_mape = np.mean(valid_mape)
                    mvalid_rmse = np.mean(valid_rmse)
                    string = str(iterNum) + " " + str(mvalid_loss) + " " + str(mvalid_rmse) + " " + str(mvalid_mape)+"\n"
                    f.writelines(string)
                iterNum = iterNum + 1
            t2 = time.time()
            train_time.append(t2 - t1)
            # validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []

            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:, 0, :, :])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            if mvalid_loss < best_valid_loss:
                # 保存最佳模型
                torch.save(engine.model.state_dict(), args.save)
                print("best model saved")
                best_valid_loss = mvalid_loss

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(
                log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
                flush=True)
        f.close()
    else:
        for i in range(1,args.epochs+1):
            #if i % 10 == 0:
                #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
                #for g in engine.optimizer.param_groups:
                    #g['lr'] = lr
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)
                trainx= trainx.transpose(1, 3) # batch*2*T*N
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)
                metrics = engine.train(trainx, trainy[:,0,:,:])
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                if iter % args.print_every == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
            t2 = time.time()
            train_time.append(t2-t1)
            #validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []


            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:,0,:,:])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            if mvalid_loss<best_valid_loss:
                # 保存最佳模型
                torch.save(engine.model.state_dict(), args.save)
                print("best model saved")
                best_valid_loss = mvalid_loss

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save))


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(best_valid_loss,4)))


    amae = []
    amape = []
    armse = []
    for i in range(3):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 3 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
