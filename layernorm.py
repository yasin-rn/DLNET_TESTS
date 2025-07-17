
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
    Main function to replicate the LayerNorm forward pass from layernorm.cu.
    """
    # --- Define dimensions ---
    model_size = 8
    seq_len = 5
    batch_size = 2
    beam_size = 2

    # --- Create and Reshape Tensor ---
    # Mimic: Tensor<float> Inputs({ BatchSize * BeamSize * SeqLen,ModelSize });
    # Mimic: Functions::FillWithCoeff(Inputs, 0.1f, 0.01f, 0.001f);
    initial_shape = (batch_size * beam_size * seq_len, model_size)
    inputs_2d = fill_with_coeff(initial_shape, 0.1, 0.01, 0.001)

    # Mimic: Tensor<float> Inputs4D = Inputs.Reshape({ BatchSize,BeamSize,SeqLen,ModelSize });
    inputs_4d = inputs_2d.reshape(batch_size, beam_size, seq_len, model_size)

    print("Inputs:\n", inputs_4d, "\n\n")

    # --- Create and Apply LayerNorm ---
    # Mimic: LayerNorm<float> LayerNorm(8);
    # In PyTorch, normalized_shape is the size of the dimension(s) to normalize.
    # Here, it's the last dimension (model_size).
    layer_norm = nn.LayerNorm(normalized_shape=model_size)

    # Mimic: auto Output = LayerNorm.Forward(Inputs4D);
    output = layer_norm(inputs_4d)

    # --- Print Output ---
    print("Output:\n", output, "\n\n")


if __name__ == "__main__":
    # Set print options for better readability
    torch.set_printoptions(precision=3, sci_mode=False)
    main()
