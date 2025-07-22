import torch
import torch.nn as nn

# Çıktı formatını daha okunaklı hale getirelim
torch.set_printoptions(precision=4, sci_mode=False)


def fill_with_coeff(shape, start_val, x_coef, y_coef):
    """
    Fills a 2D tensor with values based on their indices, mimicking the
    C++ FillWithCoeffV2 function. This ensures deterministic weight generation.
    """
    rows, cols = shape
    i_coords = torch.arange(rows, dtype=torch.float32).view(-1, 1)
    j_coords = torch.arange(cols, dtype=torch.float32)
    val1 = ((start_val * i_coords + j_coords) / start_val / 10.0)
    val2 = x_coef * torch.sin(2 * 3.14 * val1)
    val3 = y_coef * torch.sin(4 * 3.14 * val1)
    tensor = val2 + val3 + 1e-5
    return tensor


batch_size = 2
seq_len = 5
input_size = 8
hidden_size = 10

W_ih = fill_with_coeff((hidden_size, input_size), 0.1, 0.2, 0.3)
W_hh = fill_with_coeff((hidden_size, hidden_size), 0.4, 0.5, 0.6)
b_ih = torch.zeros(hidden_size)
b_hh = torch.zeros(hidden_size)

x = fill_with_coeff((batch_size*seq_len, input_size), 0.5, 0.4, 0.3)
x = x.reshape((batch_size, seq_len, input_size))

print(f"Girdi (x) boyutu: {x.shape}")
print(f"(x): {x}")
print(f"Ağırlık (W_ih) boyutu: {W_ih.shape}\n")
print(f"(W_ih):\n{W_ih}\n")

print(f"Ağırlık (W_hh) boyutu: {W_hh.shape}\n")
print(f"(W_hh):\n{W_hh}\n")


rnn_layer = nn.RNN(input_size=input_size,
                   hidden_size=hidden_size, batch_first=True)

rnn_layer.weight_ih_l0 = nn.Parameter(W_ih)
rnn_layer.weight_hh_l0 = nn.Parameter(W_hh)
rnn_layer.bias_ih_l0 = nn.Parameter(b_ih)
rnn_layer.bias_hh_l0 = nn.Parameter(b_hh)

h0_rnn = torch.zeros(1, batch_size, hidden_size)

output, h_n = rnn_layer(x, h0_rnn)
print("nn.RNN hesaplaması tamamlandı.\n")
print(output)

