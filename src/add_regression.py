from __future__ import division
import tensorflow.compat.v1 as tf
import numpy as np
import time
import os
import CRBM as crbm



class CDBN(object):
    """CONVOLUTIONAL DEEP BELIEF NETWORK"""

    def __init__(self, name, batch_size, path, data, session, verbosity=2):
        """INTENT : Initialization of a Convolutional Deep Belief Network
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        name         :        name of the CDBN
        batch_size   :        batch size to work with
        path         :        where to save and restore parameter of trained layer
        train_data   :        data to use the CDBN for training
        test_data    :        data to use the CDBN for testing
        session      :        tensorflow session (context) to use this CDBN in
        verbosity    :        verbosity of the training  (0 is low  1 is medium and 2 is high)
        ------------------------------------------------------------------------------------------------------------------------------------------
        ATTRIBUTS :
        number_layer             :        number of layer (is updated everytime add_layer() method is called
        layer_name_to_object     :        link between layer name and their corresponding crbm object
        layer_level_to_name      :        link between layer level and it name
        layer_name_to_level      :        link between layer name and it level
        input                    :        shape of the visible layer of the first layer ie where the data is to be clamped to
        fully_connected_layer    :        where the first fully connected layer occur
        locked                   :        if the CDBN model is completed ie all layer have been added
        softmax_layer            :        if the model has a softmax layer on top"""

        self.name = name
        self.batch_size = batch_size
        self.path = path + "/" + name
        tf.io.gfile.makedirs(self.path)
        self.data = data
        self.session = session
        self.verbosity = verbosity
        self.number_layer = 0
        self.layer_name_to_object = {}
        self.layer_level_to_name = {}
        self.layer_name_to_level = {}
        self.input = None
        self.fully_connected_layer = None
        self.locked = False
        self.softmax_layer = False

    def _auto_calulate_layer(self, layer_number, fully_connected):
        """INTENT : Calculate automatically the size of the input layer that we are building based on the previous layer configuration
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        layer_number          :         which layer is being built (number)
        fully_connected       :         whether the current layer is fully connected or not
        prob_maxpooling       :         whether the current layer has prob_maxpooling enabled or not
        padding               :         whether the current layer has padding enabled or not
        f_height              :         f_height of current layer
        f_width               :         f_width of current layer
        f_number              :         f_number of current layer
        ------------------------------------------------------------------------------------------------------------------------------------------
        REMARK : this works for deep layers only (not the first one) """

        previous_layer = self.layer_name_to_object[self.layer_level_to_name[layer_number - 1]]
        if not fully_connected:
            v_length = previous_layer.hidden_length / (previous_layer.prob_maxpooling + 1)
            v_channels = previous_layer.filter_number
        else:
            v_length = 1
            v_channels = (previous_layer.hidden_length / (previous_layer.prob_maxpooling + 1)) ** 2 * previous_layer.filter_number
        return int(v_length), int(v_channels)

    def add_layer(self, name, fully_connected=True, v_length="auto", v_channels="auto", f_length=1,
                  f_number=400,
                  init_biases_H=-3, init_biases_V=0.01, init_weight_stddev=0.01,
                  gaussian_unit=True, gaussian_variance=0.2,
                  prob_maxpooling=False, padding=False,
                  learning_rate=0.0001, learning_rate_decay=0.5, momentum=0.9, decay_step=50000,
                  weight_decay=0.1, sparsity_target=0.1, sparsity_coef=0.1):
        """INTENT : Add a layer to the CDBN (on the top)
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS : (same as for CRBM)
        name                  :         name of the RBM
        fully_connected       :         specify if the RBM is fully connected (True) or convolutional (False)     |   if True then obviously all height and width are 1
        v_height              :         height of the visible layer (input)
        v_width               :         width of the visible layer (input)
        v_channels            :         numbers of channels of the visible layer (input)
        f_height              :         height of the filter to apply to the visible layer
        f_width               :         width of the filter to apply to the visible layer
        f_number              :         number of filters to apply to the visible layer
        init_biases_H         :         initialization value for the bias of the hidden layer
        init_biases_V         :         initialization value for the bias of the visible layer
        init_weight_stddev    :         initialization value of the standard deviation for the kernel
        gaussian_unit         :         True if using gaussian unit for the visible layer, false if using binary unit
        gaussian_variance     :         Value of the variance of the gaussian distribution of the visible layer (only for gaussian visible unit)
        prob_maxpooling       :         True if the CRBM also include a probabilistic max pooling layer on top of the hidden layer (only for convolutional RBM)
        padding               :         True if the visible and hidden layer have same dimension (only for convolutional RBM)
        learning_rate         :     learning rate for gradient update
        learning_rate_decay   :     value of the exponential decay
        momentum              :     coefficient of the momemtum in the gradient descent
        decay_step            :     number of step before applying gradient decay
        weight_decay          :     coefficient of the weight l2 norm regularization
        sparsity_target       :     probability target of the activation of the hidden units
        sparsity_coef         :     coefficient of the sparsity regularization term
        ------------------------------------------------------------------------------------------------------------------------------------------
        REMARK : Dynamically update CDBN global view of the model"""

        try:
            if self.locked:
                raise ValueError(
                    'Trying to add layer ' + name + ' to CDBN ' + self.name + ' which has already been locked')
            if name == 'softmax_layer':
                raise ValueError(
                    'Trying to add layer ' + name + ' to CDBN ' + self.name + ' but this name is protected')
            if name in self.layer_name_to_object:
                raise ValueError(
                    'Trying to add layer ' + name + ' to CDBN ' + self.name + ' but this name is already use')
            else:
                self.layer_level_to_name[self.number_layer] = name
                self.layer_name_to_level[name] = self.number_layer

                'layer auto calculation'
                if v_length == "auto" or v_channels == "auto":
                    if self.layer_name_to_level[name] != 0:
                        if v_length == "auto"  and v_channels == "auto":
                            v_length, v_channels = self._auto_calulate_layer(self.layer_name_to_level[name],
                                                                                      fully_connected, prob_maxpooling,
                                                                                      padding, f_length, f_number)
                        else:
                            raise ValueError('You either set all 3 parameters to "auto" or none')
                    else:
                        raise ValueError('You cant set "auto" on input layer')

                if self.input is None:
                    self.input = (self.batch_size, v_length, v_channels)
                elif not fully_connected:
                    ret_out = self.layer_name_to_object[self.layer_level_to_name[self.number_layer - 1]]
                    if not (v_length == ret_out.hidden_length / (ret_out.prob_maxpooling + 1)):
                        raise ValueError(
                            'Trying to add layer ' + name + ' to CDBN ' + self.name + ' which length of visible layer does not match length of output of previous layer')
                    if not (v_channels == ret_out.filter_number):
                        raise ValueError(
                            'Trying to add layer ' + name + ' to CDBN ' + self.name + ' which number of channels of visible layer does not match number of channels of output of previous layer')
                if fully_connected and self.fully_connected_layer is None:
                    self.fully_connected_layer = self.number_layer
                self.layer_name_to_object[name] = crbm.CRBM(name, fully_connected, v_length, v_channels,
                                                            f_length, f_number,
                                                            init_biases_H, init_biases_V, init_weight_stddev,
                                                            gaussian_unit, gaussian_variance,
                                                            prob_maxpooling, padding,
                                                            self.batch_size, learning_rate, learning_rate_decay,
                                                            momentum, decay_step,
                                                            weight_decay, sparsity_target, sparsity_coef)
                self.number_layer = self.number_layer + 1
                'Where to save and restore parameter of this layer'
                tf.io.gfile.makedirs(self.path + "/" + name)

                if self.verbosity > 0:
                    print('--------------------------')
                if fully_connected:
                    message = 'Successfully adding fully connected layer ' + name + ' to CDBN ' + self.name
                    if self.verbosity > 0:
                        message += ' with has ' + str(v_channels) + ' visible units and ' + str(
                            f_number) + ' hidden units '
                else:
                    message = 'Successfully adding convolutional layer ' + name + ' to CDBN ' + self.name
                    if self.verbosity > 0:
                        message += ' with configuration of:\nVisible: (' + str(v_length) + ',' + str(v_channels) + ')\n'
                        message += 'Filters: (' + str(f_length) +  ',' + str(f_number) + ')'
                    if self.verbosity > 1 and padding:
                        message += ' with padding ON (SAME)'
                    elif padding == False:
                        message += 'with no padding and stride = 1'
                    message += '\nHidden:  (' + str(self.layer_name_to_object[name].hidden_length) + ','  + str(self.layer_name_to_object[name].filter_number) + ')'
                    if self.verbosity > 1 and prob_maxpooling:
                        message += '\nProbabilistic max pooling ON with dimension (2,2) and stride = 2: '
                    elif prob_maxpooling == False:
                        message += '\nProbabilistic max pooling OFF '
                if self.verbosity > 1 and gaussian_unit:
                    message += '\nGaussian unit ON'
                print(message)

        except ValueError as error:
            self._print_error_message(error)

    def add_regression_layer(self, learning_rate, fine_tune=False):
        """INTENT : add a regression layer on top of the CDBN
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        learning_rate         :     learning rate for gradient update
        fine_tune             :     True if fine-tuning the whole CDBN, False for layer-wise pretraining"""

        try:
            if self.locked:
                raise ValueError(
                    'Trying to add regression layer to the CDBN ' + self.name + ' which has already been locked')
            if self.regression_layer:
                raise ValueError('Trying to add regression layer to the CDBN ' + self.name + ' which already has one')
            else:
                self.regression_step = 0
                self.regression_layer = True

                ret_out = self.layer_name_to_object[self.layer_level_to_name[self.number_layer - 1]]
                self.output = int(ret_out.hidden_length / (ret_out.prob_maxpooling + 1) * ret_out.filter_number)
                with tf.variable_scope('regression_layer_cdbn'):
                    with tf.device('/cpu:0'):
                        self.W = tf.get_variable('weights_regression', (self.output, 1),  # Single node for regression
                                                 initializer=tf.truncated_normal_initializer(stddev=1 / self.output,
                                                                                             dtype=tf.float32),
                                                 dtype=tf.float32)
                        self.b = tf.get_variable('bias_regression', (1),  # Single node for regression
                                                 initializer=tf.constant_initializer(0), dtype=tf.float32)
                tf.io.gfile.makedirs(self.path + "/" + 'regression_layer')

                if self.verbosity > 0:
                    print('--------------------------')
                print('Successfully added regression layer to the CDBN ' + self.name)

                lr = tf.train.exponential_decay(learning_rate, self.regression_step, 35000, 0.1, staircase=True)
                self.regression_trainer = tf.train.MomentumOptimizer(lr, 0.9)
                self.input_placeholder = tf.placeholder(tf.float32, shape=self.input)
                eval = tf.reshape(self._get_input_level(self.number_layer, self.input_placeholder),
                                  [self.batch_size, -1])
                y = tf.matmul(eval, self.W) + self.b  # Linear activation for regression
                self.y_ = tf.placeholder(tf.float32, [None, 1])  # Placeholder for regression targets
                loss = tf.math.reduce_mean(tf.square(y - self.y_))  # Use MSE as the regression loss
                self.regression_loss = loss
                if fine_tune:
                    self.train_step = self.regression_trainer.minimize(loss)
                else:
                    (ret_w_0, ret_w_1), ret_b = self.regression_trainer.compute_gradients(loss,
                                                                                          var_list=[self.W, self.b])
                    self.train_step = self.regression_trainer.apply_gradients([(ret_w_0, ret_w_1), ret_b])
                    self.control = tf.math.reduce_mean(
                        tf.abs(tf.math.divide_no_nan(tf.multiply(ret_w_0, learning_rate), ret_w_1)))

        except ValueError as error:
            self._print_error_message(error)

    def lock_cdbn(self):
        """INTENT : lock the cdbn model"""
        try:
            if self.locked:
                raise ValueError('Trying to lock CDBN ' + self.name + ' which has already been locked')
            else:
                if not self.regression_layer:
                    ret_out = self.layer_name_to_object[self.layer_level_to_name[self.number_layer - 1]]
                    self.output = ret_out.hidden_length / (ret_out.prob_maxpooling + 1) * ret_out.filter_number
                self.locked = True

                if self.verbosity > 0:
                    print('--------------------------')
                print('Successfully locked the CDBN ' + self.name)

        except ValueError as error:
            self._print_error_message(error)

    def manage_layers(self, layers_to_pretrain, layers_to_restore, step_for_pretraining, n_for_pretraining):
        """INTENT : manage the initialization / restoration of the different layers of the CDBN
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        layers_to_pretrain        :     layers to be initialized from scratch and pretrained (names list)
        layers_to_restore         :     layers to be restored (names list)
        step_for_pretraining      :     step of training for layers to be pretrained
        n_for_pretraining         :     length of the Gibbs chain for pretraining"""

        try:
            if not self.locked:
                raise ValueError('Trying to initialize layers of CDBN ' + self.name + ' which has not been locked')
            if len(layers_to_pretrain) != 0 and ((len(layers_to_pretrain) != len(step_for_pretraining)) or (
                    len(layers_to_pretrain) != len(n_for_pretraining))):
                raise ValueError(
                    'Parameter given for the layer to be pretrained are not complete (ie 3rd and 4th argument should be list which length match one of the 1st arg)')
            else:
                self.session.run(tf.initialize_all_variables())
                for layer in layers_to_pretrain:
                    self._init_layer(layer, from_scratch=True)
                    if self.verbosity > 0:
                        print('--------------------------')
                    print('Successfully initialized the layer ' + layer + ' of CDBN ' + self.name)

                for layer in layers_to_restore:
                    self._init_layer(layer, from_scratch=False)

                for layer in layers_to_restore:
                    self._restore_layer(layer)
                if self.verbosity > 0:
                    print('--------------------------')
                print('Successfully restored the layer ' + layer + ' of CDBN ' + self.name)

            for i in range(len(layers_to_pretrain)):
                self._pretrain_layer(layers_to_pretrain[i], step_for_pretraining[i], n_for_pretraining[i])

            for i in range(len(layers_to_pretrain)):
                self._save_layer(layers_to_pretrain[i], step_for_pretraining[i])
                if self.verbosity > 0:
                    print('--------------------------')
                print('Successfully saved the layer ' + layers_to_pretrain[i] + ' of CDBN ' + self.name)

        except ValueError as error:
            self._print_error_message(error)

    import numpy as np

    import numpy as np

    def do_eval(self):
        """INTENT: Evaluate the CDBN for regression"""

        input_placeholder = tf.placeholder(tf.float32, shape=self.input)

        eval_data = tf.reshape(self._get_input_level(self.number_layer, input_placeholder), [self.batch_size, -1])
        y = tf.matmul(eval_data, self.W) + self.b
        y_ = tf.placeholder(tf.float32, [None, self.output_classes])

        # Define the regression loss function for evaluation
        loss = ...  # Define your regression loss function (e.g., Mean Absolute Error, Root Mean Squared Error, etc.)

        # Replace the data loading logic in get_next_regression_batch() with your own data loading code.
        def get_next_regression_batch():
            input_images, labels = self.data.next_batch(self.batch_size,
                                                        'test')  # Modify 'test' with appropriate dataset
            return np.reshape(input_images, self.input), labels

        num_examples = self.data.num_test_example
        steps_per_epoch = num_examples // self.batch_size
        total_loss = 0.0

        for step in range(steps_per_epoch):
            input_data, labels = get_next_regression_batch()
            loss_value = self.session.run(loss, feed_dict={input_placeholder: input_data, y_true: labels})
            total_loss += loss_value

        if self.verbosity > 0:
            print('--------------------------')
        mean_loss = total_loss / steps_per_epoch

        # You can choose to return any additional regression metrics as needed.
        return mean_loss

    def _pretrain_layer(self, rbm_layer_name, number_step, n=1):
        """INTENT: Pretrain the given layer
        -----------------------------------------------------------------------
        PARAMETERS:
        rbm_layer_name : name of CRBM layer that we want to do one step of pretraining
        number_step : number of steps to use for training
        n : length of gibbs chain to use
        """

        start = time.time()
        if self.verbosity > 0:
            start_t = time.time()
            average_cost = 0
            print('--------------------------')
        if self.verbosity == 2:
            average_control = 0
        print('Starting training the layer ' + rbm_layer_name + ' of CDBN ' + self.name)
        if self.verbosity > 0:
            print('--------------------------')
        layer_input = self.layer_name_to_object[self.layer_level_to_name[0]]
        input_placeholder = tf.placeholder(tf.float32, shape=self.input)
        step_placeholder = tf.placeholder(tf.int32, shape=(1))
        input_data = self._get_input_level(self.layer_name_to_level[rbm_layer_name], input_placeholder)
        # Define the regression loss function for pretraining
        # You can use mean squared error or any other appropriate loss function.
        loss = ...  # Define your regression loss function (e.g., mean squared error)

        # Define the optimizer for pretraining
        # optimizer = tf.train.AdamOptimizer(learning_rate)

        # Perform one step of pretraining
        a, b, c, error, control, _ = self._one_step_pretraining(rbm_layer_name, input_data, n, step_placeholder)

        # Replace the data loading logic in get_next_regression_batch() with your own data loading code.
        def get_next_regression_batch():
            input_images, _ = self.data.next_batch(self.batch_size, 'train')
            return np.reshape(input_images, self.input)

        for i in range(1, number_step):
            if self.verbosity > 0:
                start_time = time.time()
            input_images, _ = self.data.next_batch(self.batch_size, 'train')
            visible = np.reshape(input_images, self.input)
            _, _, _, err, con = self.session.run([a, b, c, error, control], feed_dict={input_placeholder: visible,
                                                                                       step_placeholder: np.array([i])})
            # Get the next batch of data for pretraining
            # visible = get_next_regression_batch()

            # Perform one step of pretraining

            if self.verbosity > 0:
                average_cost = average_cost + err
                duration = time.time() - start_time
            if self.verbosity == 2:
                average_control = average_control + con

            if self.verbosity == 1 and i % 500 == 0 and not (i % 1000 == 0):
                print(
                    'Step %d: regression loss = %.05f (%.3f sec) ----- Estimated remaining time is %.0f sec' % (
                        i, average_cost / 500, duration, (number_step - i) * (time.time() - start_t) / 500))
            elif self.verbosity == 1 and i % 1000 == 0:
                print(
                    'Step %d: regression loss = %.05f (%.3f sec) ----- Estimated remaining time is %.0f sec' % (
                        i, average_cost / 1000, duration, (number_step - i) * (time.time() - start_t) / 1000))
                average_cost = 0
                start_t = time.time()

            if self.verbosity == 2 and i % 100 == 0 and not (i % 1000 == 0):
                print(
                    'Step %d: regression loss = %.05f (%.3f sec), err = %.05f, con = %.05f and weight upgrade to weight ratio is %.2f percent  -----  Estimated remaining time is %.0f sec' % (
                        i, average_cost / (i % 1000), duration, err, con, average_control / (i % 1000) * 1,
                        (number_step - i) * (time.time() - start_t) / (i % 1000)))
            elif self.verbosity == 2 and i % 1000 == 0:
                print(
                    'Step %d: regression loss = %.05f (%.3f sec) and weight upgrade to weight ratio is %.2f percent  -----  Estimated remaining time is %.0f sec' % (
                        i, average_cost / 1000, duration, average_control / 1000 * 1,
                        (number_step - i) * (time.time() - start_t) / 1000))
                average_cost = 0
                average_control = 0
                start_t = time.time()

        if self.verbosity > 0:
            print('--------------------------')
        message = 'Successfully trained the layer ' + rbm_layer_name + ' of CDBN ' + self.name + ' in %.0f sec'
        print(message % (time.time() - start))

    def _save_layer(self, rbm_layer_name, step):
        """INTENT : Save given layer
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        rbm_layer_name         :        name of CRBM layer that we want to save
        ------------------------------------------------------------------------------------------------------------------------------------------
        REMARK : if rbm_layer_name is softmax_layer then save softmax parameter"""
        checkpoint_path = os.path.join(self.path + "/" + rbm_layer_name, 'model.ckpt')
        self.layer_name_to_object[rbm_layer_name].save_parameter(checkpoint_path, self.session, step)

    def _restore_layer(self, rbm_layer_name):
        """INTENT : Restore given layer
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        rbm_layer_name         :        name of CRBM layer that we want to restore
        ------------------------------------------------------------------------------------------------------------------------------------------
        REMARK : This method is used to restore CRBM layers only, not the softmax layer."""

        ckpt = tf.train.get_checkpoint_state(self.path + "/" + rbm_layer_name)
        return self.layer_name_to_object[rbm_layer_name].load_parameter(ckpt.model_checkpoint_path, self.session)

    def _init_layer(self, rbm_layer_name, from_scratch=True):
        """INTENT : Initialize given layer
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        rbm_layer_name         :        name of CRBM layer that we want to initialize
        from_scratch           :        if we initialize all the variable (from_scratch is True) or not """

        return self.session.run(self.layer_name_to_object[rbm_layer_name].init_parameter(from_scratch))

    def _get_input_level(self, layer_level, input_data):

        """INTENT : Get the input from the bottom to the visible layer of the given level LAYER_LEVEL
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        layer_level         :        level of the layer we need to go from bottom up to
        input_data          :        input data for the visible layer of the bottom of the cdbn"""

        ret_data = input_data
        if not layer_level == 0:
            for i in range(layer_level):
                ret_layer = self.layer_name_to_object[self.layer_level_to_name[i]]
                if ret_layer.prob_maxpooling:
                    ret_data = ret_layer.infer_probability(ret_data, method='forward', result='pooling')
                else:
                    ret_data = ret_layer.infer_probability(ret_data, method='forward', result='hidden')
                if self.fully_connected_layer == i + 1:
                    ret_data = tf.reshape(ret_data, [self.batch_size, -1])
                    # ret_data = tf.reshape(ret_data, [self.batch_size, 1, 1, ret_data.get_shape()[1].value])
                    ret_data = tf.reshape(ret_data, [self.batch_size, 1, 1, ret_data.get_shape()[1]])
        return ret_data

    def _one_step_pretraining(self, rbm_layer_name, visible_input, n, step):
        """INTENT : Do one step of contrastive divergence for the given RBM
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        rbm_layer_name         :        name of CRBM layer that we want to do one step of pretaining
        visible_input          :        configuration of the visible layer of the CRBM to train
        n                      :        length of the gibbs chain for the contrastive divergence
        step                   :        step we are at (for the learning rate decay computation)"""

        return self.layer_name_to_object[rbm_layer_name].do_contrastive_divergence(visible_input, n, step)

    def _print_error_message(self, error):
        print('----------------------------------------------')
        print('------------------ ERROR ---------------------')
        print('----------------------------------------------')
        print(error.args)
        print('----------------------------------------------')
        print('----------------------------------------------')
        print('----------------------------------------------')