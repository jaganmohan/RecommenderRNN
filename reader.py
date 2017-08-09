from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np

class Dataset(object):

    def __init__(self, users_seq, items_seq, users_seq_len,
       items_seq_len, batch_size, emb_size):
        self._users_seq = users_seq
        self._items_seq = items_seq
	self._num_users = len(users_seq)
	self._num_items = len(items_seq)
	self._batch_size = batch_size
	self._users_seq_len = users_seq_len
	self._items_seq_len = items_seq_len
	self._emb_size = emb_size
	self._max_seq_len = max(max(self._users_seq_len.values()),
		 max(self._items_seq_len.values()))
        self.users_items_in_batches = None
	self._batches = None
	self._seq_lengths = None
	self._targets = None
	self._num_batches = None
	self._u_seq_emb = None
	self._i_seq_emb = None
	self._order = None

    @property
    def __getuserseq__(self, u):
	return (self._users_seq[u], self._users_seq_len[u])

    @property
    def __getitemseq__(self, i):
	# shape of array _items_seq[i]: [sequence_length x 3]
	return (self._items_seq[i], self._items_seq_len[i])

    @property
    def __getuserembseq__(self, u):
	# list of shape: [self.emb_size+1 x sequence_length]
	return self._u_seq_emb[u]

    @property
    def __getitemembseq__(self, i):
	# list of shape: [self.emb_size+1 x sequence_length]
	return self._i_seq_emb[i]

    @property
    def batch_size(self):
	return self._batch_size

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
    def emb_size(self):
	return self._emb_size

    @property
    def max_seq_len(self):
	return self._max_seq_len

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
	users = self.users_seq.keys()
	items = self.items_seq.keys()
	u_i_batches = list()
	for x in users:
	    batch = np.zeros((2,self.batch_size), dtype=int)
	    for y in range(0,len(items),self.batch_size):
		_i = items[y:y+self.batch_size]
		batch[0] = x
		batch[1,:len(_i)] = _i
		u_i_batches.append(batch)
	return u_i_batches

    def convert_seq_to_embs(self, u_emb, i_emb):
	""" 
	Converts sequences of users and items into embeddings sequence
	"""
	user_seq_emb = collections.defaultdict(list)
	item_seq_emb = collections.defaultdict(list)
	print("[reader.py] Converting raw sequences to embedding sequences for users...")
	for u, s in self._users_seq.iteritems():
	    for i, r, ts in s:
		e = np.zeros((self.emb_size+1))
		e[:self.emb_size] = i_emb[int(i)]
		## Scale embedding with rating
		#s_emb = self.i_emb[i] * r
		#e[:self.embsize] = s_emb  
		e[self.emb_size:] = [ts]
		user_seq_emb[int(u)].append(e)
	print("[reader.py] Converting raw sequences to embedding sequence for items...")
	for i, s in self._items_seq.iteritems():
	    for u, r, ts in s:
		e = np.zeros((self.emb_size+1))
		e[:self.emb_size] = u_emb[int(u)]
		e[self.emb_size:] = [ts]
		item_seq_emb[int(i)].append(e)
	self._u_seq_emb = user_seq_emb
	self._i_seq_emb = item_seq_emb

    def prepare_batches(self, u_emb, i_emb):
        """
           will try Mini-batching sequences into chunks in next version
           https://www-i6.informatik.rwth-aachen.de/publications/download/960/Doetsch-ICFHR-2014.pdf
        """
	self.convert_seq_to_embs(u_emb, i_emb)
	self.users_items_in_batches = self.assign_u_i_tobatches()
	batches = list()
	targets = list()
	seq_lens = list()
	for count, _b in enumerate(self.u_i_in_batch,1):
	    u_seq = self.__getuserembseq__(_b[0,0])
	    _u_s, _  = self.__getuserseq__(_b[0,0])
	    _u_i_idx = np.zeros((self.batch_size,2))
	    # get the indices of user and item in respective sequences
	    # to send only that length sequence as input since further calculation
	    # of remaining sequence is not required
	    for idx,_i in enumerate(_b[1]):
		_i_s, _ = self.__getitemseq__(_i)
		_u_i_idx[idx,0] = np.where(_u_s[:,0], _i)
		_u_i_idx[idx,1] = np.where(_i_s[:,0], _b[0,0])
	    batch = np.zeros((self.batch_size, self.max_seq_len, 2, self.emb_size))
	    target = np.zeros((self.batch_size, self.max_seq_len))
	    s_len = np.zeros((self.batch_size), dtype=int)
	    for idx, _i in enumerate(_b[1]):
		m_s_len = np.max(_u_i_idx[idx])
		batch[idx,-(_u_i_idx[idx,0]+m_s_len):,0] = u_seq[:_u_i_idx[idx,0]]
		batch[idx,-(_u_i_idx[idx,1]+m_s_len):,1] = i_seq[:_u_i_idx[idx,1]]
		target[idx,-m_s_len] = _u_s[_u_i_idx[idx,0],1]
		s_len[idx] = max(m_s_len)
	    batches.append(batch.reshape(self.batch_size, self.max_seq_len, 2 * 2 * self.emb_size))
	    seq_lens.append(s_len)
	    targets.append(target)
	self._order = range(count)
	self._batches = batches
	self._seq_lengths = seq_lens
	self._targets = targets
	self._num_batches = count

    def iter_batches(self):
	""" Iterates through batches """
	for i in self.iterorder:
	    yield (self.batches[i], self.targets[i], self.seq_lengths[i])

    def batch_preprocessing(self, u_emb, i_emb):
	"""
	calls necessary prprocess methods
	Not needed when prepare_batches is called directly
	"""
	self.convert_seq_to_embs(u_emb, i_emb)
        self.users_items_in_batches = self.assign_u_i_tobatches()

    def prepare_and_iter_batches(self):
	"""
	Iterates through the batches
	"""
        for _b in self.u_i_in_batch:
            u_seq = self.__getuserembseq__(_b[0,0])
            _u_s, _  = self.__getuserseq__(_b[0,0])
            _u_i_idx = np.zeros((self.batch_size,2))
            # get the indices of user and item in respective sequences
            # to send only that length sequence as input since further calculation
            # of remaining sequence is not required
            for idx,_i in enumerate(_b[1]):
                _i_s, _ = self.__getitemseq__(_i)
                _u_i_idx[idx,0] = np.where(_u_s[:,0], _i)
                _u_i_idx[idx,1] = np.where(_i_s[:,0], _b[0,0])
	    # prepare each batch and send
            batch = np.zeros((self.batch_size, self.max_seq_len, 2, self.emb_size))
            target = np.zeros((self.batch_size, self.max_seq_len))
            s_len = np.zeros((self.batch_size), dtype=int)
            for idx, _i in enumerate(_b[1]):
                m_s_len = np.max(_u_i_idx[idx])
                batch[idx,-(_u_i_idx[idx,0]+m_s_len):,0] = u_seq[:_u_i_idx[idx,0]]
                batch[idx,-(_u_i_idx[idx,1]+m_s_len):,1] = i_seq[:_u_i_idx[idx,1]]
                target[idx,-m_s_len] = _u_s[_u_i_idx[idx,0],1]
                s_len[idx] = max(m_s_len)
            batch = batch.reshape(self.batch_size, self.max_seq_len, 2 * 2 * self.emb_size)

	    yield (batch, target, s_len)
	    
    @classmethod
    def from_path(cls, path, bs, emb_size, mode="train"):
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
            for line in fu:
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

	# Deleting unnecessary variables
	del data_users
	del data_items

	# Create and return Dataset object initialized with user and item sequence data
	settings={
	    "users_seq":users_seq,
	    "items_seq":items_seq, 
	    "users_seq_len":users_seq_len,
            "items_seq_len":items_seq_len, 
	    "batch_size":bs,
	    "emb_size":emb_size,
	}
        return cls(**settings)
