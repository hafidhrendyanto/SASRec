import sys
import copy
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from multiprocessing import queues

from sampler import DatasetSampler

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate(model, dataset, batch_size, max_sequence_length, kernel_session, mode="validation"):
    if mode not in ["test", "validation"]:
        raise Exception("mode needs to be either test or validation")
    [training_sequence, validation_sequence, test_sequence, usernum, itemnum] = copy.deepcopy(dataset)

    disounted_cumulative_gains = 0.0
    true_positives = 0.0
    validated_user = 0.0

    sampler = DatasetSampler(
        training_sequence=training_sequence,
        validation_sequence=validation_sequence,
        test_sequence=test_sequence,
        usernum=usernum,
        itemnum=itemnum,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        mode=mode,
        nworkers=16
    )

    while not sampler.check_finish():
        try:
            uid_list, input_sequences, prediction_pools = sampler.next_batch(block=False)
            score = model.predict(kernel_session, uid_list, input_sequences, prediction_pools) # (batch_size, maxlen) for both inputh_sequences and prediction_pools
            inverse_score = -score

            # argsort twice give you rank
            item_rankings = inverse_score.argsort().argsort()
            positive_item_rankings = item_rankings[:, 0]
            positive_item_rankings = np.array(positive_item_rankings)

            validated_user += len(positive_item_rankings)
            true_positives += np.sum(positive_item_rankings < 10)
            disounted_cumulative_gains += np.sum((1 / np.log2(positive_item_rankings + 2)) * (positive_item_rankings < 10))
        except queues.Empty:
            pass

    # NDCG = DCG / IDCG. 
    # Since we only have one positive sample and ideally it is ranked at the top then IDCG = # All Positive Sample = # Sequences = # Users
    normalized_discounted_cumulative_gain = disounted_cumulative_gains / validated_user 

    # Hit Rate = # True Positives / # All Positive Sample
    hit_rate = true_positives / validated_user

    return normalized_discounted_cumulative_gain, hit_rate

def original_evaluate(model, dataset, batch_size, max_sequence_length, kernel_session, mode="validation"):
    """
    Parameters
    ----------
    final_test : bool, optional
        Wheter the target is test_sequence or validation_sequence
    """
    [training_sequence, validation_sequence, test_sequence, usernum, itemnum] = copy.deepcopy(dataset)

    disounted_cumulative_gains = 0.0
    true_positives = 0.0
    validated_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for uid in users:
        if len(training_sequence[uid]) < 1 or len(validation_sequence[uid]) < 1 or len(test_sequence[uid]) < 1: continue

        # define input sequence for the current user
        input_sequence = np.zeros([max_sequence_length], dtype=np.int32)
        current_idx = max_sequence_length - 1
        if mode == "test":
            input_sequence[current_idx] = validation_sequence[uid][0]
            current_idx -= 1
        for current_item in reversed(training_sequence[uid]):
            input_sequence[current_idx] = current_item
            current_idx -= 1
            if current_idx == -1: break

        interacted_items = set(training_sequence[uid])
        interacted_items.add(0)
        if mode == "test":
            prediction_pools = [test_sequence[uid][0]]
        else:
            prediction_pools = [validation_sequence[uid][0]]
        for _ in range(100):
            sampled_negative = np.random.randint(1, itemnum + 1)
            while sampled_negative in interacted_items: sampled_negative = np.random.randint(1, itemnum + 1)
            prediction_pools.append(sampled_negative)

        predictions = -model.predict(kernel_session, [uid], [input_sequence], prediction_pools)
        predictions = predictions[0]

        # argsort twice give you rank
        positive_item_rank = predictions.argsort().argsort()[0]

        validated_user += 1

        if positive_item_rank < 10:
            disounted_cumulative_gains += 1 / np.log2(positive_item_rank + 2)
            true_positives += 1

    # NDCG = DCG / IDCG. 
    # Since we only have one positive sample and ideally it is ranked at the top then IDCG = # All Positive Sample = # Sequences = # Users
    normalized_discounted_cumulative_gain = disounted_cumulative_gains / validated_user 

    # Hit Rate = # True Positives / # All Positive Sample
    hit_rate = true_positives / validated_user

    return normalized_discounted_cumulative_gain, hit_rate
