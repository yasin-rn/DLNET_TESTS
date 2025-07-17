
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
    Main function to replicate the Multi-Head Attention forward pass.
    """
    # --- Define dimensions ---
    model_size = 8
    seq_len = 5
    batch_size = 2
    beam_size = 2
    num_head = 2

    # --- Create Input Tensor ---
    # Mimic: Tensor<float> Inputs({ BatchSize * BeamSize * SeqLen,ModelSize });
    # Mimic: Functions::FillWithCoeff(Inputs, 0.1f, 0.01f, 0.001f);
    initial_shape = (batch_size * beam_size * seq_len, model_size)
    inputs_2d = fill_with_coeff(initial_shape, 0.1, 0.01, 0.001)

    # Mimic: Tensor<float> Inputs4D = Inputs.Reshape({ BatchSize,BeamSize,SeqLen,ModelSize });
    inputs_4d = inputs_2d.reshape(batch_size, beam_size, seq_len, model_size)

    # --- Create Weight Tensors ---
    # Mimic: QW, KW, VW, OW creation
    qw = fill_with_coeff((model_size, model_size), 0.2, 0.02, 0.002)
    kw = fill_with_coeff((model_size, model_size), 0.3, 0.03, 0.003)
    vw = fill_with_coeff((model_size, model_size), 0.4, 0.04, 0.004)
    ow = fill_with_coeff((model_size, model_size), 0.5, 0.05, 0.005)

    # --- Concatenate weights for printing, as in the C++ example ---
    # Mimic: auto MHA_Weights = Functions::Concat(Weights, 0);
    mha_weights_for_print = torch.cat([qw, kw, vw, ow], dim=0)

    # --- Create and Configure MultiheadAttention Layer ---
    # Mimic: MultiheadAttention<float> MHA(...)
    mha_layer = nn.MultiheadAttention(
        embed_dim=model_size,
        num_heads=num_head,
        bias=False, # C++ code does not seem to use bias
        batch_first=True # Input format will be (N, L, E)
    )

    # --- Set Weights ---
    # Mimic: MHA.SetWeights(MHA_Weights);
    # PyTorch stores Q, K, V weights combined in 'in_proj_weight'
    # and the output projection weight in 'out_proj.weight'.
    in_proj_weight = torch.cat([qw, kw, vw], dim=0)
    with torch.no_grad():
        mha_layer.in_proj_weight.copy_(in_proj_weight)
        mha_layer.out_proj.weight.copy_(ow)

    # --- Perform Forward Pass ---
    # Prepare input for PyTorch MHA: flatten (Batch, Beam) into one dimension
    # Input shape becomes (N, L, E) -> (Batch*Beam, SeqLen, ModelSize)
    mha_input = inputs_4d.reshape(batch_size * beam_size, seq_len, model_size)

    # Mimic: auto Output = MHA.Forward(Inputs4D);
    # PyTorch MHA expects (query, key, value)
    output, _ = mha_layer(mha_input, mha_input, mha_input)

    # Reshape output back to the original 4D format
    output_4d = output.reshape(batch_size, beam_size, seq_len, model_size)

    # --- Print Results ---
    print("Inputs:\n", inputs_4d, "\n\n")
    print("MHA_Weights (Concatenated Q,K,V,O):\n", mha_weights_for_print, "\n\n")
    print("Output:\n", output_4d, "\n\n")


if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=False)
    main()
