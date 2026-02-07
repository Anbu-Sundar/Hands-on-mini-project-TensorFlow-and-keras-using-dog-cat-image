# Train model
history = model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
    callbacks=[early_stop, lr_scheduler]
)

# Fine-tuning step (optional, boosts accuracy further)
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
history_ft = model.fit(
    ds_train,
    epochs=5,
    validation_data=ds_test,
    callbacks=[early_stop, lr_scheduler]
)
