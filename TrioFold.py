import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data
from torchsummary import summary
# from FCN import FCNNet
from Network import CBAMBlock as FCNNet

from triofold.utils import *
from triofold.config import process_config
import pdb
import time
from triofold.data_generator import RNASSDataGenerator
torch.set_printoptions(threshold=np.inf)
import collections
import warnings
warnings.filterwarnings("ignore")

perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]
args = get_args()

from triofold.postprocess import postprocess_new as postprocess
from torchsummary import summary

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file
    
def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l
def get_ct_dict(predict_matrix,batch_num,ct_dict):
    
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
    return ct_dict
    
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict



class Dataset_one(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        np.set_printoptions(threshold=np.inf)
        contact, data_seq, matrix_rep, data_len, data_name, linearfold, rnafold, mxfold, contextfold, contrafold, mfold ,eternafold,spot,ufold= self.data.get_one_sample(index)
        #contact, data_seq, matrix_rep, data_len, data_name, data_pair = self.data.get_one_sample_addpairs(index)
        l = get_cut_len(data_len,80)
        #data_nc = np.zeros((2, l, l))
        data_nc = np.zeros((10, l, l))
        ensemble_list = [linearfold,rnafold,mxfold,contextfold,contrafold,mfold,eternafold,spot,ufold]
        ensemble = np.zeros((9, l, l))
        n = 0
        for i in ensemble_list:
          ensemble[n,:l,:l] = i[:l,:l]
          n+=1
        #ensemble[:l,:l] = eternafold[:l,:l]
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm_nc):
            i, j = cord
            data_nc[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_nc = data_nc.sum(axis=0).astype(np.bool)

        return contact[:l, :l], ensemble, matrix_rep, data_len, data_seq[:l], data_name, data_nc,l

        
def model_eval_all_test(contact_net,test_generator):
    device = torch.device('cuda:1')
    contact_net.train()
    result_no_train = list()
    result_no_train_fam = list()
    
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    result_nc = list()
    result_nc_tmp = list()
    ct_dict_all = {}
    dot_file_dict = {}
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []
    family_lis = []
    pred_AUC = []
    true_AUC = []
    

    predictor = 'CBAM'
        
    #for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
        nc_map_nc = nc_map.float() * contacts
        
        #if batch_n > 3:
        batch_n += 1
        #    break
        #if batch_n-1 in rep_ind:
        #    continue
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        ##seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        # matrix_reps_batch = torch.unsqueeze(
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())
        #     torch.Tensor(matrix_reps.float()).to(device), -1)

        # state_pad = torch.zeros([matrix_reps_batch.shape[0], 
        #     seq_len, seq_len]).to(device)

        # PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        tik = time.time()
        
        with torch.no_grad():
            #pred_contacts = contact_net(seq_embedding_batch,seq_embedding_batch_1)
            pred_contacts = contact_net(seq_embedding_batch).unsqueeze(0)
        
        # only post-processing without learning
        pred_numpy = pred_contacts.cpu().numpy()
        pred_numpy = pred_numpy.reshape(pred_numpy.shape[1],pred_numpy.shape[2])
        contacts_numpy = contacts_batch.cpu().numpy().reshape(pred_numpy.shape[0],pred_numpy.shape[1])
        true_AUC.append(contacts_numpy)
        pred_AUC.append(pred_numpy)

        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 50, 1, True)
        '''
        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
        '''
        

            #seq_ori, 0.01, 0.1, 100, 1.6, True) ## 1.6
        nc_no_train = nc_map.float().to(device) * u_no_train
        map_no_train = (u_no_train > 0.5).float()
        map_no_train_nc = (nc_no_train > 0.5).float()

        #map_no_train = (pred_contacts > 0.5).float()################9.12
        #map_no_train_nc = (nc_no_train > 0.5).float()
        
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)
##      add fine tune
        #threshold = 0.5
        '''
        while map_no_train.sum(axis=1).max() > 1:
            u_no_train = postprocess(u_no_train,seq_ori, 0.01, 0.1, 50, 1.0, True)
            threshold += 0.005
            map_no_train = (u_no_train > threshold).float()
        '''
        ## end fine tune
        #pdb.set_trace()
        #ct_dict_all = get_ct_dict(map_no_train,batch_n,ct_dict_all)
        ct_dict_all,dot_file_dict = get_ct_dict_fast(map_no_train,batch_n,ct_dict_all,dot_file_dict,seq_ori.cpu().squeeze(),seq_name[0])
        #ct_dict_all,dot_file_dict = get_ct_dict_fast((contacts>0.5).float(),batch_n,ct_dict_all,dot_file_dict,seq_ori.cpu().squeeze(),seq_name[0])
        #pdb.set_trace()
        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],
            contacts_batch.cpu()[i],seq_lens), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        #if seq_name[0] in fam_info:
        if True:
            result_no_train_fam += result_no_train_tmp
        
        if nc_map_nc.sum() != 0:
            #pdb.set_trace()
            result_nc_tmp = list(map(lambda i: evaluate_exact_new(map_no_train_nc.cpu()[i],
                nc_map_nc.cpu().float()[i],seq_lens), range(contacts_batch.shape[0])))
            result_nc += result_nc_tmp
            nc_name_list.append(seq_name[0])
            #if seq_lens.item() < 400 and result_nc_tmp[0][2] > 0.7:
                #pdb.set_trace()
            #    print(seq_name[0])
            #    print(result_no_train_tmp[0])
            #    print(result_nc_tmp[0])
    #pdb.set_trace()
    #print(np.mean(run_time))
    #dot_ct_file = open('results/dot_ct_file.txt','w')
    nt_exact_p,nt_exact_r,nt_exact_f1,accuracy,specificity, fpr, fnr = zip(*result_no_train)
    #pdb.set_trace()

    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))


def main():
    device  = 'cuda:1'
    test_data = RNASSDataGenerator('/home/linhb/CBAM/data/','bpRNAnew.cPickle')
    test_set = Dataset_one(test_data)
    test_generator = data.DataLoader(test_set)
    contact_net = FCNNet(channel = 9,reduction = 8)
    
    #summary(contact_net, input_size=[(9,256,256)], batch_size=4, device="cpu")
    #print(a)
    for i in range(10,11):
      MODEL_SAVED= '/model/TrioFold.pt'
    #pdb.set_trace()

      contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location='cuda:1'))

    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
      contact_net.to(device)
      model_eval_all_test(contact_net,test_generator)

    
    
    # if LOAD_MODEL and os.path.isfile(model_path):
    #     print('Loading u net model...')
    #     contact_net.load_state_dict(torch.load(model_path))
    
    
    # u_optimizer = optim.Adam(contact_net.parameters())
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs linearfold RNAfold mxfold2 contextfold contrafold mfold,eternafold,spot,ufold')
    main()






