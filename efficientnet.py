import efficientnet_builder


class EfficientNet:
    def __init__(self, data, name='efficientnet-b0'):
        self.name = name
        inputs = data['data']
        self.build(inputs)

    def build(self, inputs):
        self.layers = {}
        base = 'cls{}_fc_pose_{}'
        for c in range(1, 4):
            for p in ['xyz', 'wpqr']:
                self.layers[base.format(c,p)] = efficientnet_builder.\
                            build_model_base(inputs, self.name, True)[0]

    def load(self, sess, ignore=''):
        var_list = [x for x in tf.global_variables() if 'efficientnet' in x.name]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, tf.train.latest_checkpoint('efficient-b0/'))

