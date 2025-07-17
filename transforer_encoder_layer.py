
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
    Main function to replicate the Transformer Encoder Layer forward pass.
    """
    # --- Define dimensions ---
    model_size = 8
    seq_len = 5
    batch_size = 2
    beam_size = 2
    num_head = 2
    hidden_size = 10

    # --- Create Input Tensor ---
    initial_shape = (batch_size * beam_size * seq_len, model_size)
    inputs_2d = fill_with_coeff(initial_shape, 0.1, 0.01, 0.001)
    inputs_4d = inputs_2d.reshape(batch_size, beam_size, seq_len, model_size)

    # --- Create Weight Tensors ---
    q_w = fill_with_coeff((model_size, model_size), 0.2, 0.02, 0.002)
    k_w = fill_with_coeff((model_size, model_size), 0.3, 0.03, 0.003)
    v_w = fill_with_coeff((model_size, model_size), 0.4, 0.04, 0.004)
    o_w = fill_with_coeff((model_size, model_size), 0.5, 0.05, 0.005)
    nn_h_w = fill_with_coeff((hidden_size, model_size), 0.6, 0.06, 0.006)
    nn_o_w = fill_with_coeff((model_size, hidden_size), 0.7, 0.07, 0.007)

    # --- Create PyTorch TransformerEncoderLayer ---
    # Note: PyTorch's layer is highly optimized and combines operations.
    # We will replicate the logic step-by-step for clarity and to match the C++ code.
    encoder_layer = nn.TransformerEncoderLayer(
        model_size,
        num_head,
        dim_feedforward=hidden_size,
        dropout=0.0, # No dropout in the C++ code
        activation='relu', # As specified: DLNET_ACTIVATION_RELU
        batch_first=True, # Input format is (N, L, E)
        norm_first=False # Post-norm architecture (add&norm is after sublayer)
    )

    # --- Set Weights Manually ---
    with torch.no_grad():
        # 1. Self-Attention Weights
        # PyTorch combines Q, K, V weights. We must slice and copy.
        in_proj_weight = torch.cat([q_w, k_w, v_w], dim=0)
        encoder_layer.self_attn.in_proj_weight.copy_(in_proj_weight)
        # Output projection weight
        encoder_layer.self_attn.out_proj.weight.copy_(o_w)
        # No biases are mentioned in the C++ code for MHA
        encoder_layer.self_attn.in_proj_bias.zero_()
        encoder_layer.self_attn.out_proj.bias.zero_()

        # 2. Feedforward Network Weights
        encoder_layer.linear1.weight.copy_(nn_h_w)
        encoder_layer.linear2.weight.copy_(nn_o_w)
        # No biases are mentioned for the linear layers
        encoder_layer.linear1.bias.zero_()
        encoder_layer.linear2.bias.zero_()

        # 3. LayerNorm Weights (PyTorch defaults are 1s for weight, 0s for bias)
        # The C++ code doesn't explicitly set these, so we'll use PyTorch defaults.
        # encoder_layer.norm1.weight.fill_(1.0)
        # encoder_layer.norm1.bias.zero_()
        # encoder_layer.norm2.weight.fill_(1.0)
        # encoder_layer.norm2.bias.zero_()


    # --- Perform Forward Pass ---
    # Prepare input: flatten (Batch, Beam) into one dimension
    encoder_input = inputs_4d.reshape(batch_size * beam_size, seq_len, model_size)

    # Forward pass through the encoder layer
    encoder_out = encoder_layer(encoder_input)

    # Reshape output back to the original 4D format
    encoder_out_4d = encoder_out.reshape(batch_size, beam_size, seq_len, model_size)

    # --- Print Results ---
    print("Inputs:\n", inputs_4d, "\n\n")
    print("Encoder_Out:\n", encoder_out_4d, "\n\n")


if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=False)
    main()
