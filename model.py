from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.uid_vector = tf.placeholder(tf.int32, shape=(None)) # [B,]
        self.input_sequences = tf.placeholder(tf.int32, shape=(None, args.maxlen)) # [B, T]
        self.target_sequence = tf.placeholder(tf.int32, shape=(None, args.maxlen)) # [B, T]
        self.negative_sample = tf.placeholder(tf.int32, shape=(None, args.maxlen)) # [B, T]
        self.valid_sequences = tf.placeholder(tf.int32, shape=(None, args.maxlen)) # [B, T]
        input_mask = tf.to_float(tf.not_equal(self.input_sequences, 0))[:, :, tf.newaxis] # [B, T, 1]

        with tf.variable_scope("SASRec", reuse=reuse):
            # User Sequence Embedding, Item Embedding Table
            # [B, T, D], [vocab_size, D]
            self.current_sequence_embedding, item_embedding_table = embedding(
                self.input_sequences,
                vocab_size=itemnum + 1,
                num_units=args.hidden_units,
                zero_pad=True,
                scale=True,
                l2_reg=args.l2_emb,
                scope="input_embeddings",
                with_t=True,
                reuse=reuse
            )

            # Positional Encoding
            # [B, T, D], [T, D]
            positional_encoding, positional_encoding_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_sequences)[1]), 0), [tf.shape(self.input_sequences)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.current_sequence_embedding += positional_encoding # [B, T, D]

            # Dropout for the Input Layer
            self.current_sequence_embedding = tf.layers.dropout(
                self.current_sequence_embedding,
                rate=args.dropout_rate,
                training=tf.convert_to_tensor(self.is_training)
            )
            self.current_sequence_embedding *= input_mask # [B, T, D] * [B, T, 1] => [B, T, D]

            # Build Transformer Blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.current_sequence_embedding = multihead_attention(
                        queries=normalize(self.current_sequence_embedding),
                        keys=self.current_sequence_embedding,
                        num_units=args.hidden_units,
                        num_heads=args.num_heads,
                        dropout_rate=args.dropout_rate,
                        is_training=self.is_training,
                        causality=True,
                        scope="self_attention"
                    )

                    # Feed forward
                    self.current_sequence_embedding = feedforward(
                        normalize(self.current_sequence_embedding), 
                        num_units=[args.hidden_units, args.hidden_units],
                        dropout_rate=args.dropout_rate, 
                        is_training=self.is_training
                    )
                    self.current_sequence_embedding *= input_mask
            self.current_sequence_embedding = normalize(self.current_sequence_embedding)

        current_target_sequence = self.target_sequence # [B, T]
        current_negative_sample = self.negative_sample # [B, T]
        current_valid_sequences = self.valid_sequences # [B, T]
        current_target_sequence = tf.reshape(current_target_sequence, [tf.shape(self.input_sequences)[0] * args.maxlen]) # flatten [B, T] to [B*T,]
        current_negative_sample = tf.reshape(current_negative_sample, [tf.shape(self.input_sequences)[0] * args.maxlen]) # flatten [B, T] to [B*T,]
        current_valid_sequences = tf.reshape(current_valid_sequences, [tf.shape(self.input_sequences)[0] * args.maxlen]) # flatten [B, T] to [B*T,]
        target_embedding = tf.nn.embedding_lookup(item_embedding_table, current_target_sequence) # [B*T, D]
        negative_embedding = tf.nn.embedding_lookup(item_embedding_table, current_negative_sample) # [B*T, D]
        valid_embedding = tf.nn.embedding_lookup(item_embedding_table, current_valid_sequences) # [B*T, D]
        sequence_embedding = tf.reshape(self.current_sequence_embedding, [tf.shape(self.input_sequences)[0] * args.maxlen, args.hidden_units]) # flatten [B, T, D] to [B*T, D]

        # prediction layer
        self.positive_logits = tf.reduce_sum(target_embedding * sequence_embedding, -1) # reduce_sum([B, T, D]) => [B, T]
        self.negative_logits = tf.reduce_sum(negative_embedding * sequence_embedding, -1) # reduce_sum([B, T, D]) => [B, T]
        self.valid_logits = tf.reduce_sum(valid_embedding * sequence_embedding, -1) # reduce_sum([B, T, D]) => [B, T]

        # Create target_mast to ignore padding items (0)
        target_mask = tf.to_float(tf.not_equal(current_target_sequence, 0)) # [B*T]
        # target_mask = tf.reshape(target_mask, [tf.shape(self.input_sequences)[0] * args.maxlen]) # flatten [B, T] to [B*T,]
        self.loss = tf.reduce_sum(
            (- tf.log(tf.sigmoid(self.positive_logits) + 1e-24) - tf.log(1 - tf.sigmoid(self.negative_logits) + 1e-24)) * target_mask
        ) / tf.reduce_sum(target_mask)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(regularization_losses)

        valid_mask = tf.to_float(tf.not_equal(current_valid_sequences, 0)) # [B*T]
        self.valid_loss = tf.reduce_sum(
            (- tf.log(tf.sigmoid(self.valid_logits) + 1e-24) - tf.log(1 - tf.sigmoid(self.negative_logits) + 1e-24)) * valid_mask
        ) / tf.reduce_sum(valid_mask)
        self.valid_loss += sum(regularization_losses)
        
        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.positive_logits - self.negative_logits) + 1) / 2) * target_mask
        ) / tf.reduce_sum(target_mask)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        # Prediction Specific Kernel DAG
        self.test_item = tf.placeholder(tf.int32, shape=(None, None))
        test_item_embedding = tf.nn.embedding_lookup(item_embedding_table, self.test_item) # [B, Z, D]
        last_sequence_embedding = self.current_sequence_embedding[:, -1:, :] # [B, 1, D] | user's interest embedding for the last timestamp ;)
        self.test_logits = tf.matmul(last_sequence_embedding, tf.transpose(test_item_embedding, [0, 2, 1])) # [B, 1, D] @ [B, D, Z] => [B, 1, Z)]
        self.test_logits = self.test_logits[:, -1, :] # [B, Z]

        self.merged = tf.summary.merge_all()

    def predict(self, kernel_session, uid_list, input_sequences, prediction_pools):
        return kernel_session.run(
            self.test_logits, 
            {self.uid_vector: uid_list, self.input_sequences: input_sequences, self.test_item: prediction_pools, self.is_training: False}
        )
