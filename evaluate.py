# Evaluate
loss, acc = model.evaluate(ds_test)
print(f"\n✅ Final Accuracy: {acc*100:.2f}%")

# Save model
model.save("cat_dog_model_boosted.h5")
print("💾 Model saved as cat_dog_model_boosted.h5")
