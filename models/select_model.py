
def define_Model(opt):
    model = opt['model']



    if model in ('unipet', 'aripet'):
        from models.model_unipet import UniPETModel as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
