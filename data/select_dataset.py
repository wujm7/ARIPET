


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt.get('dataset_type', 'aripet').lower()

    if dataset_type in ['unipet', 'aripet']:
        from data.dataset_unipet import DatasetUniPET as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt.get('name', dataset_opt.get('phase', 'train'))))
    return dataset
