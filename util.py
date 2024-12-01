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

def simple_evaluate(model, dataset, batch_size, max_sequence_length, kernel_session, negative_sample_length=100, mode="validation"):
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
        negative_sample_length=negative_sample_length,
        mode=mode,
        nworkers=16
    )

    while not sampler.check_finish():
        try:
            uid_list, input_sequences, prediction_pools = sampler.next_batch(block=False)
            score = model.predict(kernel_session, uid_list, input_sequences, prediction_pools) # (batch_size, maxlen) for both inputh_sequences and prediction_pools
            score = np.array(score)
            current_batch_size = len(uid_list)

            inverse_score = -score
            item_rankings = inverse_score.argsort().argsort() # [B, T + Z] # argsort twice give you rank, Z = number of negative sample
            positive_item_rankings = item_rankings[:, max_sequence_length - 1]
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

    return normalized_discounted_cumulative_gain, hit_rate, true_positives, validated_user, disounted_cumulative_gains, validated_user

def exhaustive_evaluate(model, dataset, batch_size, max_sequence_length, kernel_session, negative_sample_length=100, mode="validation"):
    [training_sequence, validation_sequence, test_sequence, usernum, itemnum] = copy.deepcopy(dataset)

    true_positives_aggregate = np.zeros((max_sequence_length,))
    actual_positives_aggregate = np.zeros((max_sequence_length,))
    ideal_cumulative_score_aggregate = np.zeros((max_sequence_length,))
    discounted_cumulative_score_aggregate = np.zeros((max_sequence_length,))

    sampler = DatasetSampler(
        training_sequence=training_sequence,
        validation_sequence=validation_sequence,
        test_sequence=test_sequence,
        usernum=usernum,
        itemnum=itemnum,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        negative_sample_length=negative_sample_length,
        mode=mode,
        nworkers=16
    )

    while not sampler.check_finish():
        try:
            uid_list, input_sequences, prediction_pools = sampler.next_batch(block=False)
            score = model.predict(kernel_session, uid_list, input_sequences, prediction_pools) # (batch_size, maxlen) for both inputh_sequences and prediction_pools
            score = np.array(score)
            current_batch_size = len(uid_list)
            
            target_indices = np.array(prediction_pools)[:, :max_sequence_length] # [B, T]
            sequence_padding_mask = np.not_equal(target_indices, 0).astype(np.float32) # [B, T]
            sequence_positive_counts = np.sum(sequence_padding_mask, axis=-1) # [B,]. How many positive samples in each batch
            ranking_mask = np.zeros((current_batch_size, max_sequence_length + negative_sample_length))
            ranking_mask[:, :max_sequence_length] = np.where((1 - sequence_padding_mask).astype(np.bool), np.zeros(sequence_padding_mask.shape, np.float32) - float("inf"), np.zeros(sequence_padding_mask.shape, np.float32))

            score = score + ranking_mask
            item_rankings = (np.argsort(np.argsort(-score)) + 1).astype(np.float32) # [B, T + Z] | argsort twice give you rank, Z = number of negative sample
            positive_rankings = item_rankings[:, :max_sequence_length] # [B, T]

            range_size = max_sequence_length
            range_dimension = np.arange(1, range_size + 1, dtype=np.float32) # [1..K]
            current_evaluated_positive_counts = np.minimum(sequence_positive_counts[:, np.newaxis], range_dimension[np.newaxis, :]) # min(npositive, k). we calculate number of positives using it padding mask. min([B, 1], [1, K]) results in [B, K] matrix where each i, j correspond to min(npositives_i, K_j)
            found_matrix = (positive_rankings[:, :, np.newaxis] <= range_dimension[np.newaxis, np.newaxis, :]).astype(np.float32) # B, T, K matrix where T = timestamp t and K is k as in Recall@k. We're trying to determine wheter R_t <= k. R[B, T, 1] <= [1, 1, K] results in F[B, T, K] where each F_{b,t,k} correspond to wheter R_{b, t} <= k
            found_counts = np.sum(found_matrix, axis=1) # reduce_sum by the T dimension so that we get B, K. remember that each column represent a particular k so that it means we're counting all positive samples in b that have ranking lower or equal to k (i.e the true positives @ k).
        
            true_positives = np.sum(found_counts, axis=0) # reduce_sum by the B dimension, result in a vector of size [K]
            actual_positives = np.sum(current_evaluated_positive_counts, axis=0) # [K]
            true_positives_aggregate += true_positives
            actual_positives_aggregate += actual_positives

            # Calculate Discounted Cummulative Gain 
            ## Determine IDCG@k
            range_mask = (range_dimension[np.newaxis, :] <= current_evaluated_positive_counts).astype(np.float32) # [B, K]. A boolean matrix where the sum(range_mask OVER K) = sequence_positive_counts
            ideal_cumulative_score = np.cumsum(range_mask*(1. / (np.log(range_dimension + 1.) / np.log(2.))), axis=-1) # IDCG@{b, k} [B, K]. Each sequence have their own maximum IDCG based on how many positives in that sequence.
            ideal_cumulative_score_aggregate += np.sum(ideal_cumulative_score, axis=0)

            ## Calculate NDCG@k
            timestamp_discounted_scores = 1. / (np.log(positive_rankings + 1.) / np.log(2.)) # [B, T]. 1 / log_2(r + 1)
            sample_discounted_scores = np.sum(timestamp_discounted_scores[:, :, np.newaxis] * found_matrix, axis=1) # DCG@k [B, K]. sum([B, T, 1] * [B, T, K] OVER T). Think of the last dimension as selecting only the valid discounted scores for the current k. Valid values are the ones that have ranking <= k
            discounted_cumulative_score_aggregate += np.sum(sample_discounted_scores, axis=0)

        except queues.Empty:
            pass

    # NDCG = DCG / IDCG. 
    # Since we only have one positive sample and ideally it is ranked at the top then IDCG = # All Positive Sample = # Sequences = # Users
    normalized_discounted_cumulative_gain = discounted_cumulative_score_aggregate / ideal_cumulative_score_aggregate # [K]

    # Hit Rate = # True Positives / # All Positive Sample
    hit_rate = true_positives_aggregate / actual_positives_aggregate # [K]

    return normalized_discounted_cumulative_gain[10 - 1], hit_rate[10 - 1], true_positives_aggregate[10 - 1], actual_positives_aggregate[10 - 1], discounted_cumulative_score_aggregate[10 - 1], ideal_cumulative_score_aggregate[10 - 1]

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
