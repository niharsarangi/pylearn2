class DBNSampler(Block):
    """
    A Block used to sample from the last layer of a list of pre-trained RBMs.
    Here, we assume the RBM is an instance of the DBM class with only one hidden layer.
    
    It does so by computing the expected activation of the N-th RBM's hidden
    layer given the state of its visible layer, and use that expected activation
    as the state of the N+1-th RBM's visible layer.
    """
    def __init__(self, rbm_list):
        super(DBNSampler, self).__init__()
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 14)
        self.rbm_list = rbm_list

    def __call__(self, inputs):
        visible_state = inputs
        for rbm in self.rbm_list:
            # What the hidden layer sees from the visible layer
            visible_state = rbm.visible_layer.upward_state(visible_state)
            # The hidden layer's expected activation
            total_state = rbm.hidden_layers[0].mf_update(visible_state)
            # The expected activation is used as the next visible layer's state
            visible_state = rbm.hidden_layers[0].updward_state(total_state)

        # This is the last layer's expected activation
        expected_activation = visible_state

        rval = self.theano_rng.binomial(size=expected_activation.shape,
                                        p=expected_activation,
                                        dtype=expected_activation.dtype, n=1)
        return rval

    def get_input_space(self):
        return self.rbm_list[-1].visible_layer.space 

    def get_output_space(self):
        return self.rbm_list[-1].hidden_layers[-1].get_output_space()


