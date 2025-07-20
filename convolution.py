
import torch
import torch.nn as nn
import math


def fill_with_coeff(shape, start_val, x_coef, y_coef):
    """
    Fills a 2D tensor with values based on their indices, mimicking the
    C++ FillWithCoeffV2 function.
    """
    rows, cols = shape
    # Create tensors for row and column indices
    i_coords = torch.arange(rows, dtype=torch.float32).view(-1, 1)
    j_coords = torch.arange(cols, dtype=torch.float32)

    # Replicate the formula from the C++ kernel
    # T val1 = ((startVal * coord_i + coord_j) / startVal / 10);
    val1 = ((start_val * i_coords + j_coords) / start_val / 10.0)

    # T val2 = xCoef * sin(2 * 3.14 * val1);
    val2 = x_coef * torch.sin(2 * 3.14 * val1)

    # T val3 = yCoef * sin(4 * 3.14 * val1);
    val3 = y_coef * torch.sin(4 * 3.14 * val1)

    # data[i] = val2 + val3 + 1e-5;
    tensor = val2 + val3 + 1e-5
    return tensor


def main():
    """
    Main function to replicate the Convolution forward pass.
    """
    # --- Define dimensions ---
    batch_size = 1
    in_channels = 3
    out_channels = 2
    kernel_size = 4
    in_h = 8
    in_w = 8

    # --- Create Tensors using the new FillWithCoeff logic ---
    # Mimic: Tensor<float> Image2D({ BatchSize * InChannels * InH,InW });
    # Mimic: Functions::FillWithCoeff(Image2D, 0.5f, 0.4f, 0.3f);
    image_2d = fill_with_coeff(
        (batch_size * in_channels * in_h, in_w), 0.5, 0.4, 0.3)

    # Mimic: Tensor<float> Weights2D({ OutChannels * InChannels * KernelSize,KernelSize });
    # Mimic: Functions::FillWithCoeff(Weights2D, 0.1f, 0.2f, 0.3f);
    weights_2d = fill_with_coeff(
        (out_channels * in_channels * kernel_size, kernel_size), 0.1, 0.2, 0.3)

    # --- Reshape Tensors to 4D for Convolution ---
    # Mimic: Tensor<float> Image = Image2D.Reshape({ BatchSize,InChannels,InH,InW });
    image_4d = image_2d.reshape(batch_size, in_channels, in_h, in_w)

    # Mimic: Tensor<float> Weights = Weights2D.Reshape({ OutChannels,InChannels,KernelSize,KernelSize });
    weights_4d = weights_2d.reshape(
        out_channels, in_channels, kernel_size, kernel_size)

    # --- Create and Configure Convolution Layer ---
    # Mimic: Convolution<float> ConvolutinLayer(InChannels, OutChannels, KernelSize);
    # The C++ code doesn't specify padding, stride, or bias, so we use defaults
    # and set bias=False as it's not mentioned.
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=2,
        bias=False
    )

    # --- Set Weights Manually ---
    # Mimic: ConvolutinLayer.SetWeights(Weights);
    with torch.no_grad():
        conv_layer.weight.copy_(weights_4d)

    # --- Perform Forward Pass ---
    # Mimic: Tensor<float> Output = ConvolutinLayer.Forward(Image);
    output = conv_layer(image_4d)

    # --- Print Results ---
    print("Image:\n", image_4d, "\n\n")
    print("Weights:\n", weights_4d, "\n\n")
    print("Output:\n", output, "\n\n")


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    main()
