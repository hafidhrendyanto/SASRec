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

METRIC_EVALUATION = "exhaustive" # Literal[simple, exhaustive]
# METRIC_EVALUATION = "simple" # Literal[simple, exhaustive]

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
        "positional_embedding": "learnable", # Literal[none, learnable, sine]
        "metric_evaluation": METRIC_EVALUATION # Literal[simple, exhaustive]
    },
)
print(wandb.config)

if METRIC_EVALUATION == "simple":
    evaluate = simple_evaluate
elif METRIC_EVALUATION == "exhaustive":
    evaluate = exhaustive_evaluate
else:
    raise Exception()

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
kernel_session = tf.Session(config=config)

sampler = DatasetSampler(
    training_sequence=training_sequence, 
    validation_sequence=validation_sequence,
    usernum=usernum, 
    itemnum=itemnum, 
    batch_size=args.batch_size, 
    max_sequence_length=args.maxlen, 
    nworkers=16
)
model = Model(usernum, itemnum, args)
kernel_session.run(tf.initialize_all_variables())

# try:
for epoch in tqdm(range(1, args.num_epochs + 1), total=args.num_epochs, ncols=70, unit='epoch', leave=True):
    metric_aggregate = {
        "epoch/epoch": epoch - 1, # reduce by 1 to standardize with tf2's convention
        "epoch/learning_rate": learning_rate
    }

    loss_aggregate = 0.0
    valid_loss_aggregate = 0.0
    for step in range(num_batch):
        user_ids, input_sequences, target_sequence, validation_sequence, negative_sample = sampler.next_batch()
        auc, loss, valid_loss, _ = kernel_session.run(
            [model.auc, model.loss, model.valid_loss, model.train_op],
            {
                model.uid_vector: user_ids, 
                model.input_sequences: input_sequences, 
                model.target_sequence: target_sequence,
                model.negative_sample: negative_sample, 
                model.valid_sequences: validation_sequence,
                model.is_training: True
            }
        )

        loss_aggregate += float(loss)
        valid_loss_aggregate += float(valid_loss) 

    metric_aggregate["epoch/loss"] = loss_aggregate / float(num_batch)
    metric_aggregate["epoch/val_loss"] = valid_loss_aggregate / float(num_batch)

    if epoch % 5 == 0 or epoch in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        train_ndcg, train_hitrate, train_true_positives, train_actual_positives, train_dcg, train_icg  = evaluate(model, dataset, args.batch_size, args.maxlen, kernel_session, mode="train-validate")
        val_ndcg, val_hitrate, val_true_positives, val_actual_positives, val_dcg, val_icg  = evaluate(model, dataset, args.batch_size, args.maxlen, kernel_session, mode="validation")
        test_ndcg, test_hitrate, test_true_positives, test_actual_positives, test_dcg, test_icg = evaluate(model, dataset, args.batch_size, args.maxlen, kernel_session, mode="test")
        print(' | epoch:%d, valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (epoch, val_ndcg, val_hitrate, test_ndcg, test_hitrate))
        # log_file.write(str(validation_metrics) + ' ' + str(test_metrics) + '\n')
        # log_file.flush()

        metric_aggregate["epoch/true_positives@10"] = train_true_positives
        metric_aggregate["epoch/actual_positives@10"] = train_actual_positives
        metric_aggregate["epoch/hitrate@10"] = train_hitrate
        metric_aggregate["epoch/dcg@10"] = train_dcg
        metric_aggregate["epoch/icg@10"] = train_icg
        metric_aggregate["epoch/ndcg@10"] = train_ndcg

        metric_aggregate["epoch/val_true_positives@10"] = val_true_positives
        metric_aggregate["epoch/val_actual_positives@10"] = val_actual_positives
        metric_aggregate["epoch/val_hitrate@10"] = val_hitrate
        metric_aggregate["epoch/val_dcg@10"] = val_dcg
        metric_aggregate["epoch/val_icg@10"] = val_icg
        metric_aggregate["epoch/val_ndcg@10"] = val_ndcg

        metric_aggregate["epoch/test_true_positives@10"] = test_true_positives
        metric_aggregate["epoch/test_actual_positives@10"] = test_actual_positives
        metric_aggregate["epoch/test_hitrate@10"] = test_hitrate
        metric_aggregate["epoch/test_dcg@10"] = test_dcg
        metric_aggregate["epoch/test_icg@10"] = test_icg
        metric_aggregate["epoch/test_ndcg@10"] = test_ndcg

    wandb.log(metric_aggregate)
# except Exception as e:
#     sampler.close()
#     log_file.close()

#     print(e)
#     exit(1)

log_file.close()
sampler.close()
wandb.finish()
print("Done")
