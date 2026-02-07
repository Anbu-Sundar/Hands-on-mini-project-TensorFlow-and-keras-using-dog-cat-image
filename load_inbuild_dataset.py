# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
