import os
import time
import json
import argparse
import tensorflow as tf
import wandb
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# ============================================
# Parse Argument and Initialize Trackers
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
learning_rate = float(args.lr)
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

wandb.init(
    project="sasrec",
    notes="Implement Self Attentive Sequential Recommendation",
    tags=["recsys", "arXiv:1808.09781v1"],
    config={
        "architecture": "original",
        "loss": "sampled_cross_entropy", # Literal[batch_cross_entropy, sampled_cross_entropy]
        "dataset": args.dataset.replace("ml", "movielens"),
        "shuffle": False,
        "attention_block_count": int(args.num_blocks),
        "latent_dimension": int(args.hidden_units),
        "batch_size": int(args.batch_size),
        "positional_embedding": "learnable" # Literal[none, learnable, sine]
    },
)

# define custom x-axis for epoch-wise logging.
wandb.define_metric("epoch/epoch")
# set all epoch-wise metrics to be logged against epoch.
wandb.define_metric("epoch/*", step_metric="epoch/epoch")

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print 'number of batches: %.2f' % (num_batch)
print 'average sequence length: %.2f' % (cc / len(user_train))

# dump training data for archive
with open("data/user_train.json", "w") as json_file:
    json.dump(user_train, json_file)
with open("data/user_valid.json", "w") as json_file:
    json.dump(user_valid, json_file)
with open("data/user_test.json", "w") as json_file:
    json.dump(user_test, json_file)

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())

T = 0.0
t0 = time.time()

try:
    for epoch in range(1, args.num_epochs + 1):
        metric_aggregate = {
            "epoch/epoch": epoch - 1, # reduce by 1 to standardize with tf2's convention
            "epoch/learning_rate": learning_rate
        }

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})

        metric_aggregate["epoch/loss"] = float(loss)
        # TODO: add validation loss

        if epoch % 5 == 0 or epoch in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            t1 = time.time() - t0
            T += t1
            print 'Evaluating',
            t_test = evaluate(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print ''
            print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()

            # TODO: add training hitrate and ndcg
            metric_aggregate["epoch/val_hitrate@10"] = t_valid[1]
            metric_aggregate["epoch/val_ndcg@10"] = t_valid[0]
            metric_aggregate["epoch/test_hitrate@10"] = t_test[1]
            metric_aggregate["epoch/val_ndcg@10"] = t_test[0]

        wandb.log(metric_aggregate)
except:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
wandb.finish()
print("Done")
