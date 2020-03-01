from functools import partial


import tensorflow as tf

import tensorflow_io.arrow as arrow_io
from tensorflow_io.arrow.python.ops.arrow_dataset_ops import arrow_schema_to_tensor_types


class DataFrameAccessorTensorflow(object):
    def __init__(self, ml):
        self.ml = ml
        self.df = self.ml.df

    def _arrow_batch_generator(self, features, target=None, chunk_size=1024):
        column_names = features + [target] if target is not None else features
        for i1, i2, table in self.df.to_arrow_table(column_names=column_names, chunk_size=chunk_size):
            yield table.to_batches(chunk_size)[0]

    @staticmethod
    def _get_batch_arrow_schema(arrow_batch):
        output_types, output_shapes = arrow_schema_to_tensor_types(arrow_batch.schema)
        return output_types, output_shapes

    def to_dataset(self, features, target=None, chunk_size=1024, repeat=None, shuffle=False, keras=False):

        # Set up the iterator factory
        iterator_factory = partial(self._arrow_batch_generator, **{'features': features,
                                                                   'target': target,
                                                                   'chunk_size': chunk_size})
        # get the arrow schema
        output_types, output_shapes = self._get_batch_arrow_schema(next(iterator_factory()))

        # Define the TF dataset
        ds = arrow_io.ArrowStreamDataset.from_record_batches(record_batch_iter=iterator_factory(),
                                                             output_types=output_types,
                                                             output_shapes=output_shapes,
                                                             batch_mode='auto',
                                                             record_batch_iter_factory=iterator_factory)

        # Reshape the data into the appropriate format
        if keras:
            if target is not None:
                ds = ds.map(lambda *tensors: (tf.stack(tensors[:-1], axis=1), tensors[-1]))
            else:
                ds = ds.map(lambda *tensors: (tf.stack(tensors, axis=1)))
        else:
            if target is not None:
                ds = ds.map(lambda *tensors: (dict(zip(features, tensors[:-1])), tensors[-1]))
            else:
                ds = ds.map(lambda *tensors: (dict(zip(features, tensors))))

        # Repeating and shuffling the dataset if needed
        if shuffle:
            ds = ds.shuffle(chunk_size)
        if repeat is not None:
            ds = ds.repeat(repeat)

        return ds

    def make_input_function(self, features, target=None, chunk_size=1024, repeat=None, shuffle=False):
        def tf_input_function():
            return self.to_dataset(features=features,
                                   target=target,
                                   chunk_size=chunk_size,
                                   repeat=repeat,
                                   shuffle=shuffle)
        return tf_input_function


