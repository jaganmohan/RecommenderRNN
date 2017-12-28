from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np
from math import ceil
np.set_printoptions(threshold=np.nan)

class Dataset(object):

    def __init__(self, users_seq, items_seq, users_seq_len,
       items_seq_len, batch_size, chunk_size, emb_size, num_users, path):
        self._users_seq = users_seq
        self._items_seq = items_seq
	self._num_users = int(num_users)
	self._num_items = len(items_seq)
	self._batch_size = batch_size
        self._chunk_size = chunk_size
	self._users_seq_len = users_seq_len
	self._items_seq_len = items_seq_len
	self._max_seq_len = max(max(self._users_seq_len.values()),
		 max(self._items_seq_len.values()))
        self._max_seq_len = int(ceil((self._max_seq_len/chunk_size))*chunk_size)
        self.users_items_in_batches = None
	self._batches = None
	self._seq_lengths = None
	self._targets = None
	self._num_batches = None
	self._order = None
        self._path = path

    def _getuserseq(self, u):
	return self._users_seq[u], self._users_seq_len[u]

    def _getitemseq(self, i):
	return self._items_seq[i], self._items_seq_len[i]

    @property
    def batch_size(self):
	return self._batch_size

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def u_i_in_batch(self):
	return self.users_items_in_batches

    @property
    def num_users(self):
	return self._num_users

    @property
    def num_items(self):
	return self._num_items

    @property
    def max_seq_len(self):
	return self._max_seq_len

    @max_seq_len.setter
    def max_seq_len(self, _len):
        self._max_seq_len = _len

    @property
    def iterorder(self):
	if self._order == None:
	    raise BatchError("prepare_batches has not been called yet")
	return np.random.shuffle(self._order)

    @property
    def batches(self):
	if self._batches == None:
	    raise BatchError("prepare_batches has not been called yet")
	return self._batches

    @property
    def seq_lengths(self):
	if self._seq_lengths == None:
	    raise BatchError("prepare_batches has not been called yet")
	return self._seq_lengths

    @property
    def targets(self):
	if self._targets == None:
	    raise BatchError("prepare_batches has not been called yet")
	return self._targets

    def assign_u_i_tobatches(self):
	"""
	Assining users and items as a pair to batches
	shape of each batch: batch_size x 2
	"""
	print("[reader.py] Assigning user item indices to batches...")
        timeline = os.path.join(self._path,"timeline.txt")
        with open(timeline,"r") as f:
	  users = self._users_seq.keys()
	  items = self._items_seq.keys()
   	  batch = np.zeros((self._batch_size,2), dtype=int)
	  idx = 0
	  for l in f:
            u,i,r,t = l.strip().split(',')
            u,i = int(u),int(i)
            batch[idx,0] = u
	    batch[idx,1] = i
	    idx += 1
	    if idx == self._batch_size:
              yield batch
	      batch = np.zeros((self._batch_size,2), dtype=int)
	      idx = 0
	  if idx != 0:
	    yield batch

    def batch_preprocessing(self):
	"""
	calls necessary prprocess methods
	Not needed when prepare_batches is called directly
	"""
        self.users_items_in_batches = self.assign_u_i_tobatches()

    def prepare_and_iter_batches(self):
	"""
	Iterates through the batches
	"""
        # Iterates through chunks
        def iter_chunks(batch, target, s_len, max_s_lens):
            num_chunks = int(ceil(max(max_s_lens)/self.chunk_size))
            for i in range(num_chunks):
                t = i*self._chunk_size
                yield (batch[:,:,t:t+self._chunk_size,:],
                        target[:,t:t+self._chunk_size], s_len[:,i])
        # Iterates through batches
        for _b in self.assign_u_i_tobatches():
	    #print("forming batch {}".format(_b))
            batch = np.zeros((self._batch_size, 2, self._max_seq_len, 3))
            target = np.zeros((self._batch_size, self._max_seq_len))
            s_len = np.zeros((self._batch_size, 
                     int(ceil(self._max_seq_len/self._chunk_size))), dtype=int)
            max_s_lens = np.zeros((self._batch_size), dtype=int)
            # form the batch
            # get the indices of user and item in respective sequences
            # to send only that length sequence as input since further calculation
            # of remaining sequence is not required
            for idx, _cols in enumerate(_b):
              try:
                _u_s, _ = self._getuserseq(_cols[0])
                _i_s, _ = self._getitemseq(_cols[1])
                _u_idx_i = int(np.where(_u_s[:,0] == _cols[1])[0])
                _i_idx_u = int(np.where(_i_s[:,0] == _cols[0])[0])
#                print("user ",_cols[0])
#                print("item ",_cols[1])
#                print("indices ",_u_idx_i, _i_idx_u)
#                print("item_in_user ",_u_s[_u_idx_i])
#                print("user_in_item ",_i_s[_i_idx_u])
                m_s_len = max(_u_idx_i+1,_i_idx_u+1)

                batch[idx, 0, (m_s_len-_u_idx_i-1):m_s_len, :] = _u_s[:_u_idx_i+1,:]
                batch[idx, 1, (m_s_len-_i_idx_u-1):m_s_len, :] = _i_s[:_i_idx_u+1,:]
                target[idx, m_s_len-1] = _u_s[_u_idx_i,1]

                max_s_lens[idx] = m_s_len
                q,r = divmod(m_s_len,self._chunk_size)
                s_len[idx,:q] = self._chunk_size
                s_len[idx, q] = r
              except TypeError as e:
		print(e)
		print(np.where(_u_s[:,0].astype(int)==_cols[1]))
		print("Pass",_u_s[:,0],_cols[1])
              except KeyError as e:
                pass
	    yield iter_chunks(batch, target, s_len, max_s_lens)	   
 

    @classmethod
    def from_path(cls, path, bs, cs, emb_size, mode="train"):
        users_seq = dict()
        items_seq = dict()
	users_seq_len = dict()
	items_seq_len = dict()
        _path_item = os.path.join(path,"items.txt")
        _path_user = os.path.join(path,"users.txt")
        # read sequences from pre-processed data
	print("[reader.py] Reading data from pre-processed data...")
        with open(_path_item) as fi, open(_path_user) as fu:
            data_users = collections.defaultdict(list)
            data_items = collections.defaultdict(list)
            for idx,line in enumerate(fu):
                if idx == 0:
                  num_users = line.strip();
                else:
                  u, i, r, ts, n, t = line.strip().split(',')
                  data_users[u].append((t,i,r,ts))
            for line in fi:
                i, u, r, ts, n, t = line.strip().split(',')
                data_items[i].append((t,u,r,ts))
        # form array of user and item sequences  with necessary data
	print("[reader.py] Forming raw sequences for users and items...")
        for user, data in data_users.iteritems():
            users_seq[int(user)] = np.array([(i,r,ts) for t, i, r, ts in data], dtype=np.float32)
	    users_seq_len[int(user)] = len(data)
        for item, data in data_items.iteritems():
            items_seq[int(item)] = np.array([(u,r,ts) for t, u, r, ts in data], dtype=np.float32)
            items_seq_len[int(item)] = len(data)

	del data_users
	del data_items

        print(num_users)
	# Create and return Dataset object initialized with user and item sequence data
	settings={
	    "users_seq":users_seq,
	    "items_seq":items_seq, 
	    "users_seq_len":users_seq_len,
            "items_seq_len":items_seq_len, 
	    "batch_size":bs,
            "chunk_size":cs,
	    "emb_size":emb_size,
            "num_users":num_users,
            "path":path
	}
        return cls(**settings)
