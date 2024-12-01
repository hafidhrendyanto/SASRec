import math
import numpy as np
from multiprocessing import Process, Queue

def random_neqative(low, high, positive_item):
    negative_item = np.random.randint(low, high)
    while negative_item in positive_item:
        negative_item = np.random.randint(low, high)
    return negative_item

class TrainingSamplerWorker(Process):
    def __init__(self, training_sequence, usernum, itemnum, batch_size, max_sequence_length, result_queue, random_seed, *args, **kwargs):
        super(TrainingSamplerWorker, self).__init__(*args, **kwargs)
        
        self.training_sequence = training_sequence
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.result_queue = result_queue
        self.random_seed = random_seed

    def sample(self):
        """
        Sample a single sequence data for a single user
        """
        uid = np.random.randint(1, self.usernum + 1)
        while len(self.training_sequence[uid]) <= 1: uid = np.random.randint(1, self.usernum + 1)

        input_sequence = np.zeros([self.max_sequence_length], dtype=np.int32)
        target_sequence = np.zeros([self.max_sequence_length], dtype=np.int32)
        negative_sequence = np.zeros([self.max_sequence_length], dtype=np.int32)
        next_item = self.training_sequence[uid][-1]
        current_idx = self.max_sequence_length - 1 # last idx in an array of length max_sequence_length

        positive_item_set = set(self.training_sequence[uid])
        for current_item in reversed(self.training_sequence[uid][:-1]):
            input_sequence[current_idx] = current_item
            target_sequence[current_idx] = next_item
            if next_item != 0: negative_sequence[current_idx] = random_neqative(1, self.itemnum + 1, positive_item_set)
            next_item = current_item
            current_idx -= 1
            if current_idx == -1: break

        return (uid, input_sequence, target_sequence, negative_sequence)

    def run(self):
        np.random.seed(self.random_seed)
        while True:
            one_batch = []
            for i in range(self.batch_size):
                one_batch.append(self.sample())

            self.result_queue.put(zip(*one_batch))

class ValidationSamplerWorker(Process):
    def __init__(
            self, 
            training_sequence, 
            validation_sequence, 
            test_sequence, 
            assigned_users, 
            usernum, 
            itemnum, 
            batch_size, 
            max_sequence_length, 
            negative_sample_length, 
            mode, 
            result_queue,
            random_seed, 
            *args, **kwargs
        ):
        super(ValidationSamplerWorker, self).__init__(*args, **kwargs)
        
        if mode not in ["test", "validation"]:
            raise Exception("mode needs to be either test or validation")

        self.training_sequence = training_sequence
        self.validation_sequence = validation_sequence
        self.test_sequence = test_sequence
        self.assigned_users = assigned_users
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.negative_sample_length = negative_sample_length
        self.mode = mode
        self.result_queue = result_queue
        self.random_seed = random_seed

    def sample(self, uid):
        """
        Sample a single sequence data for a single user
        """
        input_sequence = np.zeros([self.max_sequence_length], dtype=np.int32)
        current_idx = self.max_sequence_length - 1
        if self.mode=="test":
            input_sequence[current_idx] = self.validation_sequence[uid][0]
            current_idx -= 1
        for current_item in reversed(self.training_sequence[uid]):
            input_sequence[current_idx] = current_item
            current_idx -= 1
            if current_idx == -1: break

        positive_item_set = set(self.training_sequence[uid])
        positive_item_set.add(0)
        if self.mode=="test":
            prediction_pool = [self.test_sequence[uid][0]]
        else:
            prediction_pool = [self.validation_sequence[uid][0]]
        for _ in range(self.negative_sample_length):
            sampled_negative = random_neqative(1, self.itemnum + 1, positive_item_set)
            prediction_pool.append(sampled_negative)

        prediction_pool = np.array(prediction_pool)

        return (uid, input_sequence, prediction_pool)

    def run(self):
        np.random.seed(self.random_seed)
        i = 0
        batch_data = []
        for uid in self.assigned_users:
            if len(self.training_sequence[uid]) < 1 or len(self.validation_sequence[uid]) < 1 or len(self.test_sequence[uid]) < 1: continue

            batch_data.append(self.sample(uid))
            i += 1

            if i % self.batch_size == 0:
                self.result_queue.put(zip(*batch_data))
                i = 0
                batch_data = []

        # exit checking
        if batch_data:
            self.result_queue.put(zip(*batch_data))


class DatasetSampler(object):
    def __init__(self, training_sequence, validation_sequence=None, test_sequence=None, usernum=1000, itemnum=1000, batch_size=64, max_sequence_length=10, mode="training", nworkers=1):
        if mode != "training" and (validation_sequence is None or test_sequence is None):
            raise Exception("If mode is not training then validation_sequence and test_sequence needs to be filled.")
        
        user_list = list(training_sequence.keys())
        partition_size = int(math.ceil(float(len(user_list)) / float(nworkers)))

        self.result_queue = Queue(maxsize = nworkers * 10)
        self.worker_pool = []
        current_partition = 0
        for i in range(nworkers):
            if mode == "training":
                new_worker = TrainingSamplerWorker(
                    training_sequence=training_sequence,
                    usernum=usernum,
                    itemnum=itemnum,
                    batch_size=batch_size,
                    max_sequence_length=max_sequence_length,
                    result_queue=self.result_queue,
                    random_seed=np.random.randint(2e9)
                )
            else:
                new_worker = ValidationSamplerWorker(
                    training_sequence=training_sequence,
                    validation_sequence=validation_sequence,
                    test_sequence=test_sequence,
                    assigned_users=user_list[current_partition*partition_size:(current_partition+1)*partition_size],
                    usernum=usernum,
                    itemnum=itemnum,
                    batch_size=batch_size,
                    max_sequence_length=max_sequence_length,
                    negative_sample_length=100,
                    mode=mode,
                    result_queue=self.result_queue,
                    random_seed=np.random.randint(2e9)
                )
                current_partition += 1

            self.worker_pool.append(new_worker)

            self.worker_pool[-1].daemon = True
            self.worker_pool[-1].start()

    def next_batch(self, block=True, timeout=None):
        return self.result_queue.get(block=block, timeout=timeout)

    def check_finish(self):
        still_running = False

        for worker in self.worker_pool:
            still_running = still_running or worker.is_alive()

        # the sampler is all finished if no worker is running and the result_queue is depleted
        return not still_running and self.result_queue.empty()

    def close(self):
        for worker in self.worker_pool:
            worker.terminate()
            worker.join()
