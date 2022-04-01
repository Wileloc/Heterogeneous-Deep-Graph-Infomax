import imp
import torch.nn as nn
from layers import HGCN, AvgReadout, Discriminator
import torch
from sklearn import cluster, utils

EPS = 1e-15

class DGI(nn.Module):
    def __init__(self, nfeat, nhid, shid, P, act):
        super(DGI, self).__init__()
        self.hgcn = HGCN(nfeat, nhid, shid, P, act)
        
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

        # self.disc = Discriminator(nhid)
        self.weight = nn.Parameter(torch.Tensor(nhid, nhid))

    def forward(self, seq1, seq2, adjs, sparse, msk, samp_bias1, samp_bias2, K, cluster_temp):
        
        h_1 = self.hgcn(seq1, adjs, sparse)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.hgcn(seq2, adjs, sparse)

        mu_init, _, _ = self.cluster_net(h_1, K, 1, 1, cluster_temp, torch.rand(K,h_1.size()[-1]))
        mu, r, dist = self.cluster_net(h_1, K, 1, 1, cluster_temp, mu_init.clone())

        return h_1, h_2, c, mu, r, dist

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h_1 = self.hgcn(seq, adjs, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()
    
    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        # print("shape", z.shape,summary.shape)
        summary = torch.unsqueeze(summary, dim=-1)
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        value = torch.squeeze(value, dim=2)
        return torch.sigmoid(value) if sigmoid else value
    
    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutal information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        
        # print('pos_loss = {}, neg_loss = {}'.format(pos_loss, neg_loss))
        # bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
        # modularity = (1./bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()
        return pos_loss + neg_loss #+ modularity

    def comm_loss(self,pos_z,mu):
        # TODO 这里不应该用unsqueeze，mu应该是一个(batch_size, cluster_number, dim)的向量
        mu = torch.unsqueeze(mu, dim=0)
        mu_red = self.read(mu, None)
        mu_summary = self.sigm(mu_red)
        return -torch.log(self.discriminate(pos_z, mu_summary, sigmoid=True) + EPS).mean()
    
    def cluster_net(self, data, k, temp, num_iter, cluster_temp, init):
        '''
        pytorch (differentiable) implementation of soft k-means clustering.
        '''
        #normalize x so it lies on the unit sphere
        data = torch.squeeze(data)
        data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
        #use kmeans++ initialization if nothing is provided
        if init is None:
            data_np = data.detach().numpy()
            norm = (data_np**2).sum(axis=1)
            init = cluster.k_means_._k_init(data_np, k, norm, utils.check_random_state(None))
            init = torch.tensor(init, requires_grad=True)
            if num_iter == 0: return init
        if torch.cuda.is_available():
            init = init.cuda()
        mu = init
        n = data.shape[0]
        d = data.shape[1]
#        data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
        for t in range(num_iter):
            #get distances between all data points and cluster centers
#            dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
            dist = data @ mu.t()
            #cluster responsibilities via softmax
            r = torch.softmax(cluster_temp*dist, 1)
            #total responsibility of each cluster
            cluster_r = r.sum(dim=0)
            #mean of points in each cluster weighted by responsibility
            cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
            #update cluster means
            new_mu = torch.diag(1/cluster_r) @ cluster_mean
            mu = new_mu
        dist = data @ mu.t()
        r = torch.softmax(cluster_temp*dist, 1)
        return mu, r, dist
