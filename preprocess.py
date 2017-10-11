from __future__ import division
from __future__ import print_function

import os
import time
import datetime
import os.path
import argparse
import collections

import numpy as np
from sklearn.preprocessing import scale

class NetflixParser(object):

    def __init__(self, path):
	self._path = path
	self._users = None
	self._train_data = None
	self._valid_data = None
	self.train_path = os.path.join(path,"training_data")
	self.valid_path = os.path.join(path,"probe.txt")

    def parse_train_data(self):
	"""parses the Netflix dataset
	Movie files available as UID,R,Time"""
	_data = list()
	train_data = dict()
	print("Aggregating training data from all movie files...")
	for _file in os.listdir(self.train_path):
	    with open(os.path.join(self.train_path,_file)) as f:
	    # First create tuples of [UID,MID,R,Time] to remove gaps from UIDs
	        for i,line in enumerate(f):
		    try:
		        if i==0:
                            mid = int(line.strip().split(':')[0])
			    continue
                        else:
			    uid, r, t = line.strip().split(',')
		    except ValueError:
		        print("Could not parse line {} ('{}') in movie file {} [ignoring]".format(i,line.strip(),f))
			continue
		    dt = datetime.datetime.strptime(t, "%Y-%m-%d")
		    ts = time.mktime(dt.timetuple())
		    _data.append((int(uid),mid,int(r),ts))
	# Standardizing timestamp data
	print("Standardizing timestamp data...")
	_data = np.array(_data)
	scale(_data[:,3], copy=False)
	# Getting all user ids from data
	self._users = set(_data[:,0].astype(int))
	for i in _data:
	    train_data[(int(i[0]),int(i[1]))] = tuple(i)
	return train_data

    def parseData(self):
	train_data = self.parse_train_data()
	# Constructing validation data
	_data = list()
	print("Getting validation data from probe.txt...")
	with open(self.valid_path) as f:
	    for i, l in enumerate(f):
		try:
		    if ":" in l:
			mid = int(l.strip().split(':')[0])
		    else:
			uid = int(l.strip())
			_data.append((uid, mid))
		except ValueError:
		    print("Could not parse line {} ('{}') in probe file [ignoring]".format(i,l.strip()))
		    continue
	print("Forming modified validation data...")
	valid_data = list()
	for i in _data:
	  if i in train_data:
	    valid_data.append(train_data[i])
	    del train_data[i]
	self._train_data = train_data.values()
	self._valid_data = valid_data


def preprocess(parser, output_dir):
	""" Pre-processes stream of [UID,MID,Rating,Timestamp] tuples
	1. Standardize timestamps
	2. Sort with respect to UIDs and relabel as consequetive sequence of UIDs
	3. Create sequences for Users and Movies
	4. Create train and validation files from the sequences
	"""
	parser.parseData()
	train_data = parser._train_data
	valid_data = parser._valid_data
	# Relabeling users to remove gaps in user IDs
	user2id = dict(zip(parser._users, range(1,len(parser._users)+1)))
        #print(user2id)
	## Creating sequences for training
	print("Creating user and item sequences for training data...")
	user_seq = collections.defaultdict(list)
	item_seq = collections.defaultdict(list)
	for uid, iid, r, ts in sorted(train_data, key=lambda x:x[3]):
	    user_seq[user2id[int(uid)]].append((int(iid),r,ts))
	    item_seq[int(iid)].append((user2id[int(uid)],r,ts))
	# Write out the sequences
	print("Writing training data user and item sequences to files...")
	train_path_item = os.path.join(output_dir,"train","items.txt")
	train_path_user = os.path.join(output_dir,"train","users.txt")
	with open(train_path_item,"w") as ti, open(train_path_user,"w") as tu:
            tu.write("{}\n".format(len(parser._users)))
	    for uid in user2id.values():
	        #t = 1
                #tu.write("{},0,0,0,1,{}\n".format(uid,t))
                t = 1
		for iid, r, ts in user_seq[uid]:
		    tu.write("{},{},{},{},0,{}\n".format(uid,iid,r,ts,t))
		    t += 1
	    for iid in item_seq.keys():
                #t = 1
                #ti.write("{},0,0,0,1,{}\n".format(iid,t))
                t = 1
		for uid, r, ts in item_seq[iid]:
		    ti.write("{},{},{},{},0,{}\n".format(iid,uid,r,ts,t))
		    t += 1 

	## Creating sequences for validation
	print("Creating user and item sequences for validation data...")
	user_seq = collections.defaultdict(list)
        item_seq = collections.defaultdict(list)
	print("valid_data ",valid_data[0])
        for uid, iid, r, ts in sorted(valid_data, key=lambda x:x[3]):
            user_seq[user2id[int(uid)]].append((int(iid),r,ts))
            item_seq[int(iid)].append((user2id[int(uid)],r,ts))
        # Write out the sequences
	print("Writing validation data user and item sequences to files...")
        valid_path_item = os.path.join(output_dir,"valid","items.txt")
        valid_path_user = os.path.join(output_dir,"valid","users.txt")
        with open(valid_path_item,"w") as vi, open(valid_path_user,"w") as vu:
            vu.write("{}\n".format(len(parser._users)))
            for uid in user2id.values():
                #t = 1
                #vu.write("{},0,0,0,1,{}\n".format(uid,t))
                t = 1
                for iid, r, ts in user_seq[uid]:
                    vu.write("{},{},{},{},0,{}\n".format(uid,iid,r,ts,t))
                    t += 1
            for iid in item_seq.keys():
                #t = 1
                #vi.write("{},0,0,0,1,{}\n".format(iid,t))
                t = 1
                for uid, r, ts in item_seq[iid]:
                    vi.write("{},{},{},{},0,{}\n".format(iid,uid,r,ts,t))
                    t += 1
	print("Done preprocessing of data")


def main(args):
	if args.which == "Netflix":
	    parser = NetflixParser(args.path)
	else:
	    raise RuntimeError("Unknown dataset!")
	preprocess(parser, args.output_dir)

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("which", choices=("Netflix"))
	parser.add_argument("path")
	parser.add_argument("--output-dir", default="./")
	return parser.parse_args()

if __name__ == '__main__':
	args = _parse_args()
	main(args)
