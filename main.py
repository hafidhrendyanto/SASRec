import os
import sys
import time
import json
import argparse
import tensorflow as tf
import wandb
from sampler import DatasetSampler
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
parser.add_argument('--original_source', action="store_true")

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
        "dataset": args.dataset.replace("ml", "movielens") + ("-mcauley" if not args.original_source else ""),
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

if args.original_source:
    print("Training using original dataset from movielens web")

    with open(r"../sasrec [Kang & McAuley, 2018]/datasets/%s/training_sequences.json" % (args.dataset), "r") as json_file:
        training_sequence = json.load(json_file)
        training_sequence = {int(uid): sequence for uid, sequence in training_sequence.items()}
    with open(r"../sasrec [Kang & McAuley, 2018]/datasets/%s/validation_sequences.json" % (args.dataset), "r") as json_file:
        validation_sequence = json.load(json_file)
        validation_sequence = {int(uid): sequence for uid, sequence in validation_sequence.items()}
    with open(r"../sasrec [Kang & McAuley, 2018]/datasets/%s/test_sequences.json" % (args.dataset), "r") as json_file:
        test_sequence = json.load(json_file)
        test_sequence = {int(uid): sequence for uid, sequence in test_sequence.items()}

    itemnum = 0
    usernum = max(training_sequence.keys())
    for sequences in [training_sequence, validation_sequence, test_sequence]:
        for sequence in sequences.values():
            itemnum = max(max(sequence), itemnum)

    dataset = [training_sequence, validation_sequence, test_sequence, usernum, itemnum]
else:
    dataset = data_partition(args.dataset)
    [training_sequence, validation_sequence, test_sequence, usernum, itemnum] = dataset

num_batch = len(training_sequence) / args.batch_size
samples = 0.0
for u in training_sequence:
    samples += len(training_sequence[u])

print(usernum, itemnum)
print('number of sequences: %d' % len(training_sequence))
print('number of batches: %d' % (num_batch))
print('average sequence length: %.2f' % (samples / len(training_sequence)))
print(training_sequence[1])

# dump training data for archive
with open("data/training_sequence.json", "w") as json_file:
    json.dump(training_sequence, json_file)
with open("data/validation_sequence.json", "w") as json_file:
    json.dump(validation_sequence, json_file)
with open("data/test_sequence.json", "w") as json_file:
    json.dump(test_sequence, json_file)

log_file = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = DatasetSampler(
    training_sequence=training_sequence, 
    usernum=usernum, 
    itemnum=itemnum, 
    batch_size=args.batch_size, 
    max_sequence_length=args.maxlen, 
    nworkers=3
)
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

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=True, unit='batch'):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                        model.is_training: True})

        metric_aggregate["epoch/loss"] = float(loss)
        # TODO: add validation loss

        if epoch % 5 == 0 or epoch in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            t1 = time.time() - t0
            T += t1
            sys.stdout.write('Evaluating')
            t_valid = evaluate(model, dataset, args.batch_size, args.maxlen, sess, mode="validation")
            t_test = evaluate(model, dataset, args.batch_size, args.maxlen, sess, mode="test")
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            log_file.write(str(t_valid) + ' ' + str(t_test) + '\n')
            log_file.flush()
            t0 = time.time()

            # TODO: add training hitrate and ndcg
            metric_aggregate["epoch/val_hitrate@10"] = t_valid[1]
            metric_aggregate["epoch/val_ndcg@10"] = t_valid[0]
            metric_aggregate["epoch/test_hitrate@10"] = t_test[1]
            metric_aggregate["epoch/test_ndcg@10"] = t_test[0]

        wandb.log(metric_aggregate)
except Exception as e:
    sampler.close()
    log_file.close()

    print(e)
    exit(1)

log_file.close()
sampler.close()
wandb.finish()
print("Done")
