**C++ Output (DLNET)**
Transformer_Output :
tensor([[[[-1.3777, -1.2936, -0.8467, 0.0173, 0.9348, 1.3381, 0.9968, 0.2310],
         [-1.1997, -1.4363, -0.9826, -0.0161, 0.8698, 1.2075, 1.0077, 0.5497],
         [-0.9240, -1.6966, -1.0637, 0.2845, 1.0978, 1.0026, 0.6715, 0.6278],
         [-0.7719, -1.5112, -1.2446, -0.1962, 0.7746, 1.1069, 0.9931, 0.8494],
         [-0.8249, -1.2879, -1.2197, -0.4799, 0.5271, 1.1804, 1.2209, 0.8838]],

        [[0.9210, 1.3390, 0.8779, 0.2506, -0.0973, -0.4291, -1.0880, -1.7742],
         [1.3997, 1.3062, 0.6996, 0.1249, -0.2715, -0.7138, -1.2121, -1.3331],
         [1.3717, 1.3238, 0.7324, 0.1307, -0.2988, -0.7429, -1.2138, -1.3031],
         [1.2833, 1.3396, 0.7842, 0.1748, -0.2751, -0.7466, -1.2444, -1.3157],
         [0.8247, 1.2649, 1.1261, 0.5294, -0.3548, -1.1769, -1.4313, -0.7821]]],

       [[[-1.3772, -1.2932, -0.8470, 0.0169, 0.9351, 1.3388, 0.9967, 0.2298],
         [-1.2026, -1.4351, -0.9806, -0.0158, 0.8694, 1.2080, 1.0084, 0.5483],
         [-0.9387, -1.6913, -1.0562, 0.2835, 1.0956, 1.0085, 0.6788, 0.6198],
         [-0.7769, -1.5109, -1.2419, -0.1941, 0.7758, 1.1082, 0.9937, 0.8462],
         [-0.8229, -1.2868, -1.2203, -0.4823, 0.5246, 1.1797, 1.2223, 0.8857]],

        [[0.9043, 1.3391, 0.8848, 0.2555, -0.0936, -0.4232, -1.0842, -1.7826],
         [1.3999, 1.3057, 0.6987, 0.1252, -0.2700, -0.7125, -1.2124, -1.3345],
         [1.3720, 1.3233, 0.7316, 0.1309, -0.2975, -0.7418, -1.2141, -1.3043],
         [1.2849, 1.3390, 0.7829, 0.1742, -0.2747, -0.7458, -1.2442, -1.3164],
         [0.8288, 1.2688, 1.1246, 0.5239, -0.3572, -1.1732, -1.4281, -0.7876]]]], dims=[2, 2, 5, 8], dtype=float32, device=cuda:0)

**Python Output (Pytorch)**
Transformer_Output:
 tensor([[[[-1.378, -1.294, -0.847,  0.017,  0.935,  1.338,  0.997,  0.231],
          [-1.200, -1.436, -0.983, -0.016,  0.870,  1.208,  1.008,  0.550],
          [-0.924, -1.697, -1.064,  0.285,  1.098,  1.003,  0.672,  0.628],
          [-0.772, -1.511, -1.245, -0.196,  0.775,  1.107,  0.993,  0.849],
          [-0.825, -1.288, -1.220, -0.480,  0.527,  1.180,  1.221,  0.884]],

         [[ 0.921,  1.339,  0.878,  0.251, -0.097, -0.429, -1.088, -1.774],
          [ 1.400,  1.306,  0.700,  0.125, -0.271, -0.714, -1.212, -1.333],
          [ 1.372,  1.324,  0.732,  0.131, -0.299, -0.743, -1.214, -1.303],
          [ 1.283,  1.340,  0.784,  0.175, -0.275, -0.747, -1.244, -1.316],
          [ 0.825,  1.265,  1.126,  0.529, -0.355, -1.177, -1.431, -0.782]]],


        [[[-1.377, -1.293, -0.847,  0.017,  0.935,  1.339,  0.997,  0.230],
          [-1.203, -1.435, -0.981, -0.016,  0.869,  1.208,  1.008,  0.548],
          [-0.939, -1.691, -1.056,  0.284,  1.096,  1.008,  0.679,  0.620],
          [-0.777, -1.511, -1.242, -0.194,  0.776,  1.108,  0.994,  0.846],
          [-0.823, -1.287, -1.220, -0.482,  0.525,  1.180,  1.222,  0.886]],

         [[ 0.904,  1.339,  0.885,  0.256, -0.094, -0.423, -1.084, -1.783],
          [ 1.400,  1.306,  0.699,  0.125, -0.270, -0.713, -1.212, -1.335],
          [ 1.372,  1.323,  0.732,  0.131, -0.298, -0.742, -1.214, -1.304],
          [ 1.285,  1.339,  0.783,  0.174, -0.275, -0.746, -1.244, -1.316],
          [ 0.829,  1.269,  1.125,  0.524, -0.357, -1.173, -1.428, -0.788]]]],
       grad_fn=<ViewBackward0>)