import logging

import pytest
pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_io")

import tensorflow as tf

import vaex
import vaex.ml


logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# Set up the data to be used in this test file
df = vaex.ml.datasets.load_iris()
df_train, df_test = df.ml.train_test_split(test_size=0.2, verbose=False)
features = df.column_names[:4]
target = 'class_'


def test_to_dataset_tensorflow():
    ds = df_test.ml.tensorflow.to_dataset(features=features, target=target, chunk_size=10)
    list_ds = list(ds)
    assert len(list_ds) == 3  # The number of "batches" in the iterable, as defined by the chunk_size arg

    idx_min = 0
    idx_max = 10
    for batch in list_ds:
        assert len(batch) == 2
        assert batch[1].numpy().tolist() == df_test[target][idx_min:idx_max].tolist()
        for feat in batch[0]:
            assert batch[0][feat].numpy().tolist() == df_test[feat][idx_min:idx_max].tolist()

        idx_min += 10
        idx_max += 10


def test_to_dataset_keras():
    ds = df_test.ml.tensorflow.to_dataset(features=features, target=target, chunk_size=10, as_dict=False)
    list_ds = list(ds)
    assert len(list_ds) == 3  # The number of "batches" in the iterable, as defined by the chunk_size arg

    idx_min = 0
    idx_max = 10
    for batch in list_ds:
        assert len(batch) == 2  # check that each batch has feature matrix and target matrix (hence tuple of len 2)
        # Assert the target
        assert batch[1].numpy().tolist() == df_test[target][idx_min:idx_max].tolist()
        # Assert the feature tensor
        assert batch[0].numpy().tolist() == df_test[features][idx_min:idx_max].values.tolist()
        # Shift index according to what the batch should be
        idx_min += 10
        idx_max += 10


def test_make_input_function():
    train_fn = df_train.ml.tensorflow.make_input_function(features=features, target=target, chunk_size=120, repeat=5)
    eval_fn = df_test.ml.tensorflow.make_input_function(features=features, target=target, chunk_size=120)
    test_fn = df_test.ml.tensorflow.make_input_function(features=features, chunk_size=120)

    feature_columns = []
    for feat in features:
        feature_columns.append(tf.feature_column.numeric_column(key=feat))

    est = tf.estimator.LinearClassifier(n_classes=3, feature_columns=feature_columns)
    est.train(train_fn)
    evaluation = est.evaluate(eval_fn)
    assert len(evaluation) == 4  # the exact contents of the dict is not important, important is that it can be obrained

    predictions = list(est.predict(test_fn, yield_single_examples=False))[0]
    assert predictions['logits'].shape == (30, 3)
    assert predictions['probabilities'].shape == (30, 3)
    assert predictions['classes'].shape == (30, 1)
    assert predictions['class_ids'].shape == (30, 1)
    assert predictions['all_class_ids'].shape == (30, 3)
    assert predictions['all_classes'].shape == (30, 3)
    assert list(predictions.keys()) == ['logits', 'probabilities', 'class_ids', 'classes', 'all_class_ids', 'all_classes']


@pytest.mark.parametrize("repeat", [1, 3, 10])
@pytest.mark.parametrize("shuffle", [False, True])
def test_make_input_function_options(repeat, shuffle):
    num_batches = repeat * 3

    input_function = df_test.ml.tensorflow.make_input_function(features=features,
                                                               target=target,
                                                               chunk_size=10,
                                                               repeat=repeat,
                                                               shuffle=shuffle)
    ds = input_function()
    list_ds = list(ds)
    assert len(list_ds) == num_batches  # The number of "batches" in the iterable, as defined by the chunk_size arg


# def test_to_dataset_keras_model_train():

#     # Define the keras input functions
#     train_ds = df_train.ml.tensorflow.to_dataset(features=features, target=target, keras=True)
#     test_ds = df_test.ml.tensorflow.to_dataset(features=features, keras=True)

#     # Defne a simple keras model
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Dense(units=4, input_shape=(4,), activation='relu'))
#     model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#     model.compile(optimizer=tf.keras.optimizers.RMSprop(),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     model.fit(train_ds)
#     predictions = model.predict(test_ds)
#     assert len(predictions) == len(df_test)
