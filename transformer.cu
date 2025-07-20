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

	Tensor<float> Input2D({ BatchSize * BeamSize * SeqLen,ModelSize });

	Tensor<float> SelfAttn_WQ({ ModelSize,ModelSize });
	Tensor<float> SelfAttn_WK({ ModelSize,ModelSize });
	Tensor<float> SelfAttn_WV({ ModelSize,ModelSize });
	Tensor<float> SelfAttn_WO({ ModelSize,ModelSize });


	Tensor<float> MultiheadAttn_WQ({ ModelSize,ModelSize });
	Tensor<float> MultiheadAttn_WK({ ModelSize,ModelSize });
	Tensor<float> MultiheadAttn_WV({ ModelSize,ModelSize });
	Tensor<float> MultiheadAttn_WO({ ModelSize,ModelSize });

	Tensor<float> Linear1_W({ HiddenSize,ModelSize });
	Tensor<float> Linear2_W({ ModelSize,HiddenSize });

	Functions::FillWithCoeff(Input2D, 5.0f, 4.0f, 3.0f);
	std::cout << "Input2D :\n" << Input2D.ToString() << "\n\n";

	Tensor<float> Input = Input2D.Reshape({ BatchSize,BeamSize,SeqLen,ModelSize });

	Functions::FillWithCoeff(SelfAttn_WQ, 0.1f, 0.2f, 0.3f);
	Functions::FillWithCoeff(SelfAttn_WK, 0.2f, 0.3f, 0.4f);
	Functions::FillWithCoeff(SelfAttn_WV, 0.3f, 0.4f, 0.5f);
	Functions::FillWithCoeff(SelfAttn_WO, 0.4f, 0.5f, 0.6f);

	Functions::FillWithCoeff(MultiheadAttn_WQ, 0.8f, 0.7f, 0.6f);
	Functions::FillWithCoeff(MultiheadAttn_WK, 0.7f, 0.6f, 0.5f);
	Functions::FillWithCoeff(MultiheadAttn_WV, 0.6f, 0.5f, 0.4f);
	Functions::FillWithCoeff(MultiheadAttn_WO, 0.5f, 0.4f, 0.3f);

	std::vector<Tensor<float>> SelfAttn_Weights_List = { SelfAttn_WQ,SelfAttn_WK,SelfAttn_WV,SelfAttn_WO };
	std::vector<Tensor<float>> MultiheadAttn_Weights_List = { MultiheadAttn_WQ,MultiheadAttn_WK,MultiheadAttn_WV,MultiheadAttn_WO };

	auto SelfAttn_Weights = Functions::Concat(SelfAttn_Weights_List, 0);
	auto MultiheadAttn_Weights = Functions::Concat(MultiheadAttn_Weights_List, 0);


	Functions::FillWithCoeff(Linear1_W, 0.6f, 0.7f, 0.8f);
	Functions::FillWithCoeff(Linear2_W, 0.7f, 0.8f, 0.9f);
	std::cout << "SelfAttn_WQ :\n" << SelfAttn_WQ.ToString() << "\n\n";
	std::cout << "SelfAttn_WK :\n" << SelfAttn_WK.ToString() << "\n\n";
	std::cout << "SelfAttn_WV :\n" << SelfAttn_WV.ToString() << "\n\n";
	std::cout << "SelfAttn_WO :\n" << SelfAttn_WO.ToString() << "\n\n";

	Transformer<float> Transformer(ModelSize, SeqLen, HiddenSize, NumOfHead, DLNET_ACTIVATION_RELU, 1, 1);

	for (size_t i = 0; i < Transformer.DecoderLayers.Layers.size(); i++)
	{
		Transformer.EncoderLayers.Layers[i].SetMhaWeights(SelfAttn_Weights);
		Transformer.EncoderLayers.Layers[i].SetLinear1Weights(Linear1_W);
		Transformer.EncoderLayers.Layers[i].SetLinear2Weights(Linear2_W);

		Transformer.DecoderLayers.Layers[i].SetMaskedMhaWeights(SelfAttn_Weights);
		Transformer.DecoderLayers.Layers[i].SetMhaWeights(MultiheadAttn_Weights);
		Transformer.DecoderLayers.Layers[i].SetLinear1Weights(Linear1_W);
		Transformer.DecoderLayers.Layers[i].SetLinear2Weights(Linear2_W);
	}

	auto Transformer_Output = Transformer.Forward(Input, Input);

	std::cout << "Transformer_Output :\n" << Transformer_Output.ToString() << "\n\n";


	cudnnDestroy(cudnnHandle);

	return 0;
}