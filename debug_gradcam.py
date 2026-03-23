import tensorflow as tf
import numpy as np

MODEL_PATH = "model/dr_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
img_batch = np.random.rand(1, 128, 128, 3).astype('float32')

last_conv_layer_name = 'last_conv_layer'

# Reconstruct for GradCAM
x = model.inputs[0]
last_conv_output = None
for layer in model.layers:
    x = layer(x)
    if layer.name == last_conv_layer_name:
        last_conv_output = x

grad_model = tf.keras.models.Model(model.inputs, [last_conv_output, x])

try:
    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        last_conv_layer_output, preds = grad_model(img_batch)
        class_channel = preds[:, 0]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    print("Gradient shape:", grads.shape)
    print("GradCAM successful!")
except Exception as e:
    print("Error:", e)
