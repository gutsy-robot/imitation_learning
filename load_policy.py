import pickle, tensorflow as tf, tf_util, numpy as np


def load_policy(filename):
    print("loading policy from file: ", filename)
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    print("data type of expert policy is: ", type(data))
    # print(data.items())
    # assert len(data.keys()) == 2

    # specifies the type of non-linearity function used.
    nonlin_type = data['nonlin_type']

    print("nonlintype is: ", nonlin_type)
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

    print("policy_type is: ", policy_type)
    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    print("policy params datatype is: ", type(policy_params))
    print(policy_params['obsnorm'].keys())

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

    # Keep track of input and output dims (i.e. observation and action dims) for the user

    def build_policy(obs_bo):
        def read_layer(l):
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

        def apply_nonlin(x):
            if nonlin_type == 'lrelu':
                return tf_util.lrelu(x, leak=.01)  # openai/imitation nn.py:233
            elif nonlin_type == 'tanh':
                return tf.tanh(x)
            else:
                raise NotImplementedError(nonlin_type)

        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']

        # useful for calculating standard deviation.
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']

        print("mean[0] shape  is: ", obsnorm_mean[0].shape)
        # print("mean sq is: ", obsnorm_meansq[0])

        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))

        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)

        normedobs_bo = (obs_bo - obsnorm_mean) / (
                    obsnorm_stdev + 1e-6)  # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation

        curr_activations_bd = normedobs_bo

        # Hidden layers next
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            curr_activations_bd = apply_nonlin(tf.matmul(curr_activations_bd, W) + b)

        # Output layer
        W, b = read_layer(policy_params['out'])
        output_bo = tf.matmul(curr_activations_bd, W) + b
        print("type of output_bo is: ", type(output_bo))
        print(output_bo.shape)
        return output_bo

    obs_bo = tf.placeholder(tf.float32, [None, None])
    a_ba = build_policy(obs_bo)
    policy_fn = tf_util.function([obs_bo], a_ba)
    print("type returned by load policy is: ", type(policy_fn))
    return policy_fn
