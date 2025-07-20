
import torch
import torch.nn as nn

def fill_with_coeff(shape, start_val, x_coef, y_coef):
    """
    Fills a 2D tensor with values based on their indices, mimicking the
    C++ FillWithCoeff function.
    """
    rows, cols = shape
    i_vals = torch.arange(rows, dtype=torch.float32).view(-1, 1) * x_coef
    j_vals = torch.arange(cols, dtype=torch.float32) * y_coef
    tensor = start_val + i_vals + j_vals
    return tensor

def main():
    """
    Main function to replicate the Transformer Decoder Layer forward pass.
    """
    # --- Define dimensions ---
    model_size = 8
    seq_len = 5
    batch_size = 2
    beam_size = 2
    num_head = 2
    hidden_size = 10

    # --- Create Input Tensors ---
    # Decoder's own input
    input_2d = fill_with_coeff((batch_size * beam_size * seq_len, model_size), 5.0, 4.0, 3.0)
    input_4d = input_2d.reshape(batch_size, beam_size, seq_len, model_size)
    # Output from the encoder
    encoder_output_2d = fill_with_coeff((batch_size * beam_size * seq_len, model_size), 6.0, 5.0, 4.0)
    encoder_output_4d = encoder_output_2d.reshape(batch_size, beam_size, seq_len, model_size)

    # --- Create Weight Tensors ---
    # 1. For Masked Self-Attention
    self_attn_wq = fill_with_coeff((model_size, model_size), 0.1, 0.2, 0.3)
    self_attn_wk = fill_with_coeff((model_size, model_size), 0.2, 0.3, 0.4)
    self_attn_wv = fill_with_coeff((model_size, model_size), 0.3, 0.4, 0.5)
    self_attn_wo = fill_with_coeff((model_size, model_size), 0.4, 0.5, 0.6)

    # 2. For Cross-Attention (Encoder-Decoder Attention)
    cross_attn_wq = fill_with_coeff((model_size, model_size), 0.8, 0.7, 0.6)
    cross_attn_wk = fill_with_coeff((model_size, model_size), 0.7, 0.6, 0.5)
    cross_attn_wv = fill_with_coeff((model_size, model_size), 0.6, 0.5, 0.4)
    cross_attn_wo = fill_with_coeff((model_size, model_size), 0.5, 0.4, 0.3)

    # 3. For Feed-Forward Network
    linear1_w = fill_with_coeff((hidden_size, model_size), 0.6, 0.7, 0.8)
    linear2_w = fill_with_coeff((model_size, hidden_size), 0.7, 0.8, 0.9)

    # --- Create PyTorch TransformerDecoderLayer ---
    # Using positional arguments for d_model and nhead based on previous findings
    decoder_layer = nn.TransformerDecoderLayer(
        model_size,
        num_head,
        dim_feedforward=hidden_size,
        dropout=0.0,
        activation='relu',
        batch_first=True,
        norm_first=False
    )

    # --- Set Weights Manually ---
    with torch.no_grad():
        # 1. Set Self-Attention (Masked MHA) weights
        self_attn_in_proj = torch.cat([self_attn_wq, self_attn_wk, self_attn_wv], dim=0)
        decoder_layer.self_attn.in_proj_weight.copy_(self_attn_in_proj)
        decoder_layer.self_attn.out_proj.weight.copy_(self_attn_wo)
        decoder_layer.self_attn.in_proj_bias.zero_()
        decoder_layer.self_attn.out_proj.bias.zero_()

        # 2. Set Cross-Attention (MHA) weights
        cross_attn_in_proj = torch.cat([cross_attn_wq, cross_attn_wk, cross_attn_wv], dim=0)
        decoder_layer.multihead_attn.in_proj_weight.copy_(cross_attn_in_proj)
        decoder_layer.multihead_attn.out_proj.weight.copy_(cross_attn_wo)
        decoder_layer.multihead_attn.in_proj_bias.zero_()
        decoder_layer.multihead_attn.out_proj.bias.zero_()

        # 3. Set Feed-Forward weights
        decoder_layer.linear1.weight.copy_(linear1_w)
        decoder_layer.linear2.weight.copy_(linear2_w)
        decoder_layer.linear1.bias.zero_()
        decoder_layer.linear2.bias.zero_()

        # 4. Set LayerNorm weights to default (1s for weight, 0s for bias)
        decoder_layer.norm1.weight.fill_(1.0)
        decoder_layer.norm1.bias.zero_()
        decoder_layer.norm2.weight.fill_(1.0)
        decoder_layer.norm2.bias.zero_()
        decoder_layer.norm3.weight.fill_(1.0)
        decoder_layer.norm3.bias.zero_()

    # --- Perform Forward Pass ---
    # Prepare inputs: flatten (Batch, Beam) into one dimension
    tgt = input_4d.reshape(batch_size * beam_size, seq_len, model_size)
    memory = encoder_output_4d.reshape(batch_size * beam_size, seq_len, model_size)

    # Create the causal mask for the self-attention mechanism
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

    # Forward pass through the decoder layer
    decoder_out = decoder_layer(tgt, memory, tgt_mask=tgt_mask)

    # Reshape output back to the original 4D format
    decoder_out_4d = decoder_out.reshape(batch_size, beam_size, seq_len, model_size)

    # --- Print Results ---
    print("Input (tgt):\n", input_4d, "\n\n")
    print("Encoder Output (memory):\n", encoder_output_4d, "\n\n")
    print("Decoder_Out:\n", decoder_out_4d, "\n\n")


if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=False)
    main()
