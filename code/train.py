import numpy as np

### Main File ###

def train_net():
    # Load libraries / unpack config files

    # Build Network

    ### Train model ###
    print '--- training network'


    # Start training loop
    epoch = 0
    step_idx = 0
    val_record = []
    while epoch < n_epochs:
        epoch = epoch + 1
        
        if config['shuffle']:
            np.random.shuffle(minibatch_range)

        if config['resume_train'] and epoch == 1:
            load_epoch = config['load_epoch']
            load_weights(layers, config['weights_dir'], load_epoch)
            lr_to_load = np.load(config['weights_dir'] + 'lr_' + str(load_epoch) + '.npy')
            val_record = list(np.load(config['weights_dir'] + 'val_record.npy'))
            learning_rate.set_value(lr_to_load)
            load_momentums(vels, config['weights_dir'], load_epoch)
            epoch = load_epoch + 1
            
        if flag_para_load:
            load_send_quene.put()
            load_send_quene.put()
            load_send_quene.put()

        # Run over minibatch_range
        count = 0
        for minibatch_index in minibatch_range:
            num_iter = (epoch - 1) * n_train_batches + count
            count = count + 1
            if count == 1:
                s = time.time()
            if count == 20:
                e = time.time()
                print "time per 20 iter: ", (e - s)

            cost_ij = train_model_wrap(train_model, shared_x
                                       shared_y, rand_arr, img_mean,
                                       count, minibatch_index,
                                       minibatch_range, batch_size,
                                       train_filenames, train_labels,
                                       flag_para_load,
                                       config['batch_crop_mirror'],
                                       send_queue = load_send_queue,
                                       recv_queue = load_recv_queue
                                   )
            if num_iter % config['print_freq'] == 0:
                print '', num_iter
                print '', cost_ij
                if config['print_train_error']:
                    print 'training error rate: ', train_error()
                    
            if flag_para_load and (count < len(minibatch_range)):
                load_send_queue.put('calc_finished')
        # Test on Validation Set
        print '--- testing on validation set'
        #
        DropoutLayer.SetDropoutOff()

        this_validation_error, this_validation_loss = get_val_error_loss(
            rand_arr, shared_x, shared_y,
            val_filenames, val_labels,
            flag_para_load, img_mean,
            batch_size, validate_model,
            send_queue=load_send_queue, recv_queue=load_recv_queue
        )

        print('epoch %i: validation loss %f ' %
              (epoch, this_validation_loss))
        print('epoch %i: validation error %f %%' %
              (epoch, this_validation_error * 100.))

        val_record.append([this_validation_error, this_validation_loss])
        np.save(config['weights_dir'] + 'val_record.npy', val_record)

        DropoutLayer.SetDropoutOn()
        # Adapt Learning Rate
        step_id = adjust_learning_rate(config, epoch, step_id,
                                       val_record, learning_rate)
        
        # Save Weights
        if epoch % config['snapshot_freq'] == 0:
            save_weights(layers, config['weights_dir'], epoch)

    print('Optimization complete.')

def main():
    train_net()

if __name__ = "__main__":
    main()    
