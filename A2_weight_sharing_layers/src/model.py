import tensorflow as tf
from tensorflow.keras import layers

def rotate_kernel_90(kernel, k):
    """
    Rotate the spatial dimensions (H, W) of each (in_channel, out_channel) kernel slice.

    Args:
        kernel: 4D tensor of shape (H, W, in_channels, out_channels)
        k: number of 90° rotations (an integer from 0 to 3)
    Returns:
        rotated_tensor: a tensor with the same shape as kernel, rotated spatially by k*90°.
    """
    # kernel.shape returns a tuple (H, W, in_channels, out_channels)
    H, W, in_channels, out_channels = kernel.shape

    rotated = []
    # Loop over input channels
    for i in range(in_channels):
        slices_per_channel = []
        # Loop over output channels
        for j in range(out_channels):
            # Extract the (H, W) slice for input channel i and output channel j
            slice_ij = kernel[:, :, i, j]
            # tf.image.rot90 requires a 3D input; expand dims and rotate
            rotated_slice = tf.image.rot90(tf.expand_dims(slice_ij, axis=-1), k=k)
            # Remove the singleton dimension, recovering (H, W)
            rotated_slice = tf.squeeze(rotated_slice, axis=-1)
            slices_per_channel.append(rotated_slice)
        # Stack along the output channel axis for the i-th input channel
        rotated.append(tf.stack(slices_per_channel, axis=-1))
    # Stack along the input channel axis to restore original order.
    rotated_tensor = tf.stack(rotated, axis=2)
    # Shape: (H, W, in_channels, out_channels)
    return rotated_tensor

class GConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='SAME', **kwargs):
        """
        A custom group convolutional layer that is equivariant to 90° rotations.
        
        Args:
            filters: Number of output filters per rotation (total effective output channels = filters * 4)
            kernel_size: Size of the (square) convolution kernel.
            strides: Convolution stride.
            padding: Padding mode ('SAME' or 'VALID').
        """
        super(GConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        # input_shape: (batch, height, width, channels)
        in_channels = int(input_shape[-1])
        kernel_shape = (self.kernel_size, self.kernel_size, in_channels, self.filters)
        
        # Define a base kernel (for the unrotated orientation)
        self.kernel = self.add_weight(
            shape=kernel_shape, 
            initializer='glorot_uniform', 
            trainable=True,
            name='kernel'
        )
        # Define a bias for each rotation (total output channels = filters * 4)
        self.bias = self.add_weight(
            shape=(self.filters * 4,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super(GConv2D, self).build(input_shape)
    
    def call(self, inputs):
        # Generate rotated versions of the base kernel for 0°, 90°, 180°, and 270°
        rotated_kernels = [rotate_kernel_90(self.kernel, k=r) for r in range(4)]
        
        conv_outputs = []
        for rotated_kernel in rotated_kernels:
            conv = tf.nn.conv2d(
                inputs,
                rotated_kernel,
                strides=[1, self.strides, self.strides, 1],
                padding=self.padding
            )
            conv_outputs.append(conv)
        
        # Concatenate along the channel axis: output shape (batch, H, W, filters * 4)
        output = tf.concat(conv_outputs, axis=-1)
        output = tf.nn.bias_add(output, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        # For 'SAME' padding, spatial dimensions are preserved.
        batch_size, height, width, _ = input_shape
        return (batch_size, height, width, self.filters * 4)

class GMaxPooling(layers.Layer):
    def call(self, inputs):
        # inputs should have shape: (batch, H, W, filters*4)
        # Separate the rotation dimension. Assume that the channel dimension is a multiple of 4.
        input_shape = tf.shape(inputs)
        num_filters = inputs.shape[-1] // 4  # Base filter count
        # Reshape to (batch, H, W, num_filters, 4)
        reshaped = tf.reshape(inputs, (-1, input_shape[1], input_shape[2], num_filters, 4))
        # Apply max-pooling along the rotation dimension
        pooled = tf.reduce_max(reshaped, axis=-1)
        # Pooled output shape: (batch, H, W, num_filters)
        return pooled

    def compute_output_shape(self, input_shape):
        num_filters = input_shape[-1] // 4
        return (input_shape[0], input_shape[1], input_shape[2], num_filters)
