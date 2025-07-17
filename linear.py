

import torch
import torch.nn as nn

def fill_with_coeff(shape, start_val, x_coef, y_coef):
    """
    Fills a 2D tensor with values based on their indices, mimicking the
    C++ FillWithCoeff function.
    Value at (i, j) = start_val + (i * x_coef) + (j * y_coef)
    """
    rows, cols = shape
    i_vals = torch.arange(rows, dtype=torch.float32).view(-1, 1) * x_coef
    j_vals = torch.arange(cols, dtype=torch.float32) * y_coef
    tensor = start_val + i_vals + j_vals
    return tensor

def main():
    """
    Main function to replicate the linear layer forward pass from linear.cu.
    """
    # --- Define dimensions ---
    in_features = 8
    out_features = 4
    batch_size = 8

    # --- Create Tensors ---
    # Mimic: Tensor<float> Inputs({ BatchSize,InFeatures });
    # Mimic: Functions::FillWithCoeff(Inputs, 0.1f, 0.01f, 0.001f);
    inputs = fill_with_coeff((batch_size, in_features), 0.1, 0.01, 0.001)

    # Mimic: Tensor<float> Weights({ OutFeatures,InFeatures });
    # Mimic: Functions::FillWithCoeff(Weights, 0.2f, 0.02f, 0.002f);
    weights = fill_with_coeff((out_features, in_features), 0.2, 0.02, 0.002)

    # Mimic: Tensor<float> Bias({ 1,OutFeatures });
    # Mimic: Functions::FillWithCoeff(Bias, 0.3f, 0.03f, 0.003f);
    # Note: The C++ code creates a 2D bias, but PyTorch uses a 1D bias vector.
    bias = fill_with_coeff((1, out_features), 0.3, 0.03, 0.003).squeeze(0)

    # --- Print Tensors ---
    print("Inputs:\n", inputs, "\n\n")
    print("Weights:\n", weights, "\n\n")
    print("Bias:\n", bias, "\n\n")

    # --- Create and configure Linear Layer ---
    # Mimic: Linear<float> LinearF(InFeatures, OutFeatures, true);
    linear_layer = nn.Linear(in_features, out_features, bias=True)

    # Mimic: LinearF.SetWeights(Weights, Bias);
    with torch.no_grad(): # We are setting weights manually, not training
        linear_layer.weight.copy_(weights)
        linear_layer.bias.copy_(bias)

    # --- Perform Forward Pass ---
    # Mimic: auto Output = LinearF.Forward(Inputs);
    output = linear_layer(inputs)

    # --- Print Output ---
    print("Output:\n", output, "\n\n")


if __name__ == "__main__":
    # Set print options for better readability
    torch.set_printoptions(precision=3, sci_mode=False)
    main()

