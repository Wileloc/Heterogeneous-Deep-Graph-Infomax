import numpy as np
import scipy.sparse as sp
import pickle

author_idx_map = {}
conf_idx_map = {}
paper_idx_map = {}
term_idx_map = {}


def text2map(f, mapdict):
    for i, l in enumerate(f.readlines()):
        l = l.replace("\n", "")
        idx, item = l.split("\t")
        if item not in mapdict:
            mapdict[int(idx)] = i


def build_edge(f, edgelist, start_map_dict, end_map_dict):
    for i, l in enumerate(f.readlines()):
        l = l.replace("\n", "")
        start, end = l.split("\t")
        if int(start) in start_map_dict and int(end) in end_map_dict:
            if not [start_map_dict[int(start)], end_map_dict[int(end)]] in edgelist:
                edgelist.append([start_map_dict[int(start)], end_map_dict[int(end)]])


with open('author.txt') as f:
    text2map(f, author_idx_map)
with open('conf.txt') as f:
    text2map(f, conf_idx_map)
with open('paper.txt', encoding='gbk') as f:
    text2map(f, paper_idx_map)
with open('term.txt') as f:
    text2map(f, term_idx_map)

paper_author_edges = []
paper_conf_edges = []
paper_term_edges = []

with open('paper_author.txt') as f:
    print('build paper_author edges')
    build_edge(f, paper_author_edges, paper_idx_map, author_idx_map)
with open('paper_conf.txt') as f:
    print('build paper_conf edges')
    build_edge(f, paper_conf_edges, paper_idx_map, conf_idx_map)
with open('paper_term.txt') as f:
    print('build paper_term edges')
    build_edge(f, paper_term_edges, paper_idx_map, term_idx_map)

paper_author_edges = np.array(paper_author_edges)
paper_conf_edges = np.array(paper_conf_edges)
paper_term_edges = np.array(paper_term_edges)

paper_author_adj = sp.coo_matrix(
    (np.ones(paper_author_edges.shape[0]), (paper_author_edges[:, 0], paper_author_edges[:, 1])),
    shape=(len(paper_idx_map), len(author_idx_map)), dtype=np.int32)

paper_conf_adj = sp.coo_matrix(
    (np.ones(paper_conf_edges.shape[0]), (paper_conf_edges[:, 0], paper_conf_edges[:, 1])),
    shape=(len(paper_idx_map), len(conf_idx_map)), dtype=np.int32)

paper_term_adj = sp.coo_matrix(
    (np.ones(paper_term_edges.shape[0]), (paper_term_edges[:, 0], paper_term_edges[:, 1])),
    shape=(len(paper_idx_map), len(term_idx_map)), dtype=np.int32)

author_paper_author_adj = sp.coo_matrix.dot(paper_author_adj.transpose(), paper_author_adj)
#movie_actor_movie_adj = movie_actor_movie_adj.todense()
paper_conf_paper_adj = sp.coo_matrix.dot(paper_conf_adj, paper_conf_adj.transpose())
author_paper_conf_paper_author_adj = sp.coo_matrix.dot(paper_author_adj.transpose(), sp.coo_matrix.dot(paper_conf_paper_adj, paper_author_adj))
#movie_director_movie_adj = movie_director_movie_adj.todense()
paper_term_paper_adj = sp.coo_matrix.dot(paper_term_adj, paper_term_adj.transpose())
author_paper_term_paper_author_adj = sp.coo_matrix.dot(paper_author_adj.transpose(), sp.coo_matrix.dot(paper_term_paper_adj, paper_author_adj))
##movie_keyword_movie_adj = movie_keyword_movie_adj.todense()

matrix_temp = np.ones(author_paper_author_adj.shape)-np.eye(author_paper_author_adj.shape[0])
author_paper_author_adj = author_paper_author_adj.multiply(matrix_temp)
#movie_actor_movie_adj = movie_actor_movie_adj.todense()
author_paper_conf_paper_author_adj = author_paper_conf_paper_author_adj.multiply(matrix_temp)
#movie_director_movie_adj = movie_director_movie_adj.todense()
author_paper_term_paper_author_adj = author_paper_term_paper_author_adj.multiply(matrix_temp)
#movie_keyword_movie_adj = movie_keyword_movie_adj.todense()
with open('net_APA_adj.pickle', 'wb') as m:
    pickle.dump(author_paper_author_adj, m)
with open('net_APCPA_adj.pickle', 'wb') as a:
    pickle.dump(author_paper_conf_paper_author_adj, a)
with open('net_APTPA_adj.pickle', 'wb') as d:
    pickle.dump(author_paper_term_paper_author_adj, d)

