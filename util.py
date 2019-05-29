import torch
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset


class SiameseNetworkDataset(Dataset):

    def __init__(self, a, b, c, co):
        self.a = a
        self.b = b
        self.c = c
        self.co = co

    def __getitem__(self, index):
        return self.a[index], self.b[index], self.c[index], self.co[index]

    def __len__(self):
        return self.a.size()[0]


def init_weights(net):
    con_w1 = ['cnn1.0.weight', 'cnn3.2.weight']
    con_w2 = ['cnn3.0.weight', 'cnn1.2.weight']
    norm_w = ['cnn1.1.weight', 'cnn1.3.weight', 'cnn3.1.weight', 'cnn3.3.weight']
    fc14_w = ['fc1.0.weight', 'fc4.0.weight']
    fc23_w = ['fc2.0.weight', 'fc3.0.weight']
    fc_bias = ['fc1.0.bias', 'fc2.0.bias', 'fc3.0.bias', 'fc4.0.bias']
    for name, params in net.named_parameters():
        if name in con_w1:
            init.xavier_normal_(params)
        elif name in con_w2:
            init.xavier_normal_(params)
        elif name in norm_w:
            params.data.normal_()
        elif name in fc14_w:
            init.xavier_normal_(params)
        elif name in fc23_w:
            init.xavier_normal_(params)
        elif name.find('bias') != -1:
            if name in fc_bias:
                params.data.fill_(1)
            else:
                params.data.fill_(0)


def calculate_metric(num_large, N):
    if num_large >= N:
        return 0, 0
    elif num_large == 0:
        return 1, 1
    else:
        return 1, (N - num_large) / N


# evaluate with cosine similarity based on represented vectors
def tes_vec(h_a, h_b, anchor_train, anchor_test, N, n_b):
    lens = len(anchor_test)
    anchor_a_list = anchor_test
    anchor_b_list = anchor_test
    known_b_list = anchor_train
    test_user_b = list(set(n_b)-set(known_b_list))
    vec_test_b = h_b[test_user_b]
    index, PatN, MatN = 0, 0.0, 0.0
    while index < lens:
        an_a = torch.unsqueeze(h_a[anchor_a_list[index]], dim=0)
        an_b = torch.unsqueeze(h_b[anchor_b_list[index]], dim=0)
        an_sim = F.cosine_similarity(an_a, an_b).item()
        un_an_sim = F.cosine_similarity(an_a, vec_test_b)
        larger_than_anchor = un_an_sim >= an_sim
        num_large_than_anchor = int(larger_than_anchor.sum().item())
        patN, matN = calculate_metric(num_large_than_anchor, N)
        PatN += patN
        MatN += matN
        index += 1
    PatN_t, MatN_t = PatN/lens, MatN/lens
    return PatN_t, MatN_t


# evaluate with probability based on classification
def val_classifier(h_a, h_b, anchor_train, anchor_test, paras, n_b, classifier_net):
    cuda = paras.gpu_id
    N = paras.N
    classifier_net.eval()
    lens = len(anchor_test)
    anchor_a_list = anchor_test
    anchor_b_list = anchor_test
    known_b_list = anchor_train
    test_user_b = list(set(n_b) - set(known_b_list))
    vec_test_b = h_b[test_user_b]
    index, PatN, MatN = 0, 0.0, 0.0
    while index < lens:
        an_a1 = anchor_a_list[index]
        an_a = torch.unsqueeze(h_a[an_a1], dim=0)
        an_b1 = anchor_b_list[index]
        an_b = torch.unsqueeze(h_b[an_b1], dim=0)
        an_pair_vec = torch.cat((an_a, an_b), dim=1)
        an_pair_out = classifier_net(an_pair_vec)
        an_pair_out = F.softmax(an_pair_out, dim=1)
        an_pair_pro = an_pair_out.detach()[:, 1].item()
        tmp = torch.ones(len(test_user_b), 1).to(device=cuda)
        an_ass = an_a * tmp
        un_anchor_pairs_vec = torch.cat((an_ass, vec_test_b), dim=1)
        un_anchor_pairs_out = classifier_net(un_anchor_pairs_vec)
        un_anchor_pairs_out = F.softmax(un_anchor_pairs_out, dim=1)
        un_an_pair_pro = un_anchor_pairs_out.detach()[:, 1]
        larger_than_anchor = an_pair_pro <= un_an_pair_pro
        num_large_than_anchor = int(larger_than_anchor.sum().item())
        patN, matN = calculate_metric(num_large_than_anchor, N)
        PatN += patN
        MatN += matN
        index += 1
    PatN_t, MatN_t = PatN / lens, MatN / lens
    return PatN_t, MatN_t
