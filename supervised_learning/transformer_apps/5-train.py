#!/usr/bin/env python3
"""
5 train a transformer
"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Correct transformer learning rate schedule formula
        arg1 = tf.math.rsqrt(step)  # 1/sqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)  # step * warmup_steps^(-1.5)
        
        lr = tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)
        
        return lr
    
    def get_config(self):
        return {
            "d_model": int(self.d_model),
            "warmup_steps": int(self.warmup_steps)
        }


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    creates and trains a transformer model for machine
    translation of Portuguese to English
    """
    dataset = Dataset(batch_size, max_len)
    transformer = Transformer(N, dm, h, hidden, 
                             dataset.tokenizer_pt.vocab_size + 2,
                             dataset.tokenizer_en.vocab_size + 2,
                             max_len, max_len)
    
    # Custom training setup with corrected learning rate
    lr_schedule = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='loss')
    tr_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    tr_loss = tf.keras.metrics.Mean(name='train_loss')
    
    # Create checkpoint manager
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    # Restore latest checkpoint if exists
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")
    
    # Custom training loop
    for epoch in range(epochs):
        num_batches = 0
        tr_accuracy.reset_state()
        tr_loss.reset_state()

        for batch, (inp, tar) in enumerate(dataset.data_train):
            # Target input and output for teacher forcing
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp, tar_inp)
            
            with tf.GradientTape() as tape:
                predictions = transformer(inp, tar_inp, training=True,
                                          encoder_mask=enc_padding_mask,
                                          look_ahead_mask=combined_mask,
                                          decoder_mask=dec_padding_mask)
                loss = loss_fn(tar_real, predictions)
            
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            
            tr_loss(loss)
            tr_accuracy(tar_real, predictions)
            num_batches += 1
            
            if batch % 50 == 0:
                current_lr = lr_schedule(optimizer.iterations)
                print(f'Epoch {epoch + 1}, Batch {batch}: '
                      f'Loss {tr_loss.result():.4f} '
                      f'Accuracy {tr_accuracy.result():.4f}')

        ckpt_save_path = ckpt_manager.save()
        print(f'Epoch {epoch + 1}: '
              f'Loss {tr_loss.result():.4f} '
              f'Accuracy {tr_accuracy.result():.4f}')

    return transformer