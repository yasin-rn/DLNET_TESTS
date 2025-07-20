
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
    Main function to replicate the full Transformer model forward pass.
    """
    # --- Define dimensions ---
    model_size = 8
    seq_len = 5
    batch_size = 2
    beam_size = 2
    num_head = 2
    hidden_size = 10
    num_encoder_layers = 1
    num_decoder_layers = 1

    # --- Create Input Tensor ---
    # In the C++ code, the same input is used for both encoder and decoder
    input_2d = fill_with_coeff((batch_size * beam_size * seq_len, model_size), 5.0, 4.0, 3.0)
    input_4d = input_2d.reshape(batch_size, beam_size, seq_len, model_size)

    # --- Create Weight Tensors ---
    # C++ code reuses weights for encoder's self-attn and decoder's masked self-attn
    self_attn_wq = fill_with_coeff((model_size, model_size), 0.1, 0.2, 0.3)
    self_attn_wk = fill_with_coeff((model_size, model_size), 0.2, 0.3, 0.4)
    self_attn_wv = fill_with_coeff((model_size, model_size), 0.3, 0.4, 0.5)
    self_attn_wo = fill_with_coeff((model_size, model_size), 0.4, 0.5, 0.6)

    # Weights for the decoder's cross-attention
    cross_attn_wq = fill_with_coeff((model_size, model_size), 0.8, 0.7, 0.6)
    cross_attn_wk = fill_with_coeff((model_size, model_size), 0.7, 0.6, 0.5)
    cross_attn_wv = fill_with_coeff((model_size, model_size), 0.6, 0.5, 0.4)
    cross_attn_wo = fill_with_coeff((model_size, model_size), 0.5, 0.4, 0.3)

    # C++ code reuses weights for all feed-forward networks
    linear1_w = fill_with_coeff((hidden_size, model_size), 0.6, 0.7, 0.8)
    linear2_w = fill_with_coeff((model_size, hidden_size), 0.7, 0.8, 0.9)

    # --- Create PyTorch Transformer Model ---
    transformer_model = nn.Transformer(
        d_model=model_size,
        nhead=num_head,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=hidden_size,
        dropout=0.0,
        activation='relu',
        batch_first=True
    )

    # --- Set Weights Manually ---
    with torch.no_grad():
        # 1. Set Encoder Layer Weights
        encoder_layer = transformer_model.encoder.layers[0]
        enc_in_proj = torch.cat([self_attn_wq, self_attn_wk, self_attn_wv], dim=0)
        encoder_layer.self_attn.in_proj_weight.copy_(enc_in_proj)
        encoder_layer.self_attn.out_proj.weight.copy_(self_attn_wo)
        encoder_layer.linear1.weight.copy_(linear1_w)
        encoder_layer.linear2.weight.copy_(linear2_w)
        # Zero out biases
        encoder_layer.self_attn.in_proj_bias.zero_()
        encoder_layer.self_attn.out_proj.bias.zero_()
        encoder_layer.linear1.bias.zero_()
        encoder_layer.linear2.bias.zero_()

        # 2. Set Decoder Layer Weights
        decoder_layer = transformer_model.decoder.layers[0]
        # Masked Self-Attention
        dec_self_in_proj = torch.cat([self_attn_wq, self_attn_wk, self_attn_wv], dim=0)
        decoder_layer.self_attn.in_proj_weight.copy_(dec_self_in_proj)
        decoder_layer.self_attn.out_proj.weight.copy_(self_attn_wo)
        # Cross-Attention
        dec_cross_in_proj = torch.cat([cross_attn_wq, cross_attn_wk, cross_attn_wv], dim=0)
        decoder_layer.multihead_attn.in_proj_weight.copy_(dec_cross_in_proj)
        decoder_layer.multihead_attn.out_proj.weight.copy_(cross_attn_wo)
        # Feed-Forward
        decoder_layer.linear1.weight.copy_(linear1_w)
        decoder_layer.linear2.weight.copy_(linear2_w)
        # Zero out biases
        decoder_layer.self_attn.in_proj_bias.zero_()
        decoder_layer.self_attn.out_proj.bias.zero_()
        decoder_layer.multihead_attn.in_proj_bias.zero_()
        decoder_layer.multihead_attn.out_proj.bias.zero_()
        decoder_layer.linear1.bias.zero_()
        decoder_layer.linear2.bias.zero_()


    # --- Perform Forward Pass ---
    # Prepare inputs: flatten (Batch, Beam) into one dimension
    flat_input = input_4d.reshape(batch_size * beam_size, seq_len, model_size)
    src = flat_input
    tgt = flat_input # Using the same input for src and tgt as in the C++ code

    # Create the causal mask for the decoder's self-attention
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

    # Forward pass through the full transformer model
    transformer_output = transformer_model(src, tgt, tgt_mask=tgt_mask)

    # Reshape output back to the original 4D format
    transformer_output_4d = transformer_output.reshape(batch_size, beam_size, seq_len, model_size)

    # --- Print Results ---
    print("Input (src and tgt):\n", input_4d, "\n\n")
    print("Transformer_Output:\n", transformer_output_4d, "\n\n")


if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=False)
    main()
