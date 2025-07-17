
int main()
{
	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);

	int BatchSize = 2;
	int BeamSize = 2;
	int SeqLen = 5;
	int ModelSize = 8;
	int HiddenSize = 10;
	int NumOfHead = 2;
	int DK = ModelSize / NumOfHead;
	double Scale = 1 / sqrt(static_cast<double>(DK));

	Tensor<float> Inputs({ BatchSize * BeamSize * SeqLen,ModelSize });
	Functions::FillWithCoeff(Inputs, 0.1f, 0.01f, 0.001f);

	Tensor<float> Inputs4D = Inputs.Reshape({ BatchSize,BeamSize,SeqLen,ModelSize });

	Tensor<float> Q_W({ ModelSize,ModelSize });
	Tensor<float> K_W({ ModelSize,ModelSize });
	Tensor<float> V_W({ ModelSize,ModelSize });
	Tensor<float> O_W({ ModelSize,ModelSize });
	Tensor<float> NN_H_W({ HiddenSize,ModelSize });
	Tensor<float> NN_O_W({ ModelSize,HiddenSize });


	Functions::FillWithCoeff(Q_W, 0.2f, 0.02f, 0.002f);
	Functions::FillWithCoeff(K_W, 0.3f, 0.03f, 0.003f);
	Functions::FillWithCoeff(V_W, 0.4f, 0.04f, 0.004f);
	Functions::FillWithCoeff(O_W, 0.5f, 0.05f, 0.005f);

	Functions::FillWithCoeff(NN_H_W, 0.6f, 0.06f, 0.006f);
	Functions::FillWithCoeff(NN_O_W, 0.7f, 0.07f, 0.007f);

	std::vector<Tensor<float>> Weights = { Q_W,K_W,V_W,O_W };

	auto MHA_Weights = Functions::Concat(Weights, 0);

	TransformerEncoderLayer<float> EncoderLayer(ModelSize, SeqLen, HiddenSize, NumOfHead, DLNET_ACTIVATION_RELU);

	EncoderLayer.SetMhaWeights(MHA_Weights);
	EncoderLayer.SetLinear1Weights(NN_H_W);
	EncoderLayer.SetLinear2Weights(NN_O_W);

	auto Encoder_Out = EncoderLayer.Forward(Inputs4D);

	std::cout << "Inputs :\n" << Inputs4D.ToString() << "\n\n";
	std::cout << "Encoder_Out :\n" << Encoder_Out.ToString() << "\n\n";


	cudnnDestroy(cudnnHandle);

	return 0;
}