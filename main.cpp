#include <iostream>
#include <vector>

using f32 = float;
using i32 = int;

auto activate(f32 x) -> f32 {
    return 1.f / (1.f + exp(-x));
}

struct Neuron {
    f32 value, beta;
};

class WeightMatrix {
public:
    explicit WeightMatrix(i32 outCount, i32 inCount) : inCount_(inCount), outCount_(outCount),
                                                       weights_(inCount * outCount) {
        for (auto &w: weights_) {
            w = (random() /  (float)RAND_MAX) * 0.5f;
        }
    }

public:
    auto setWeight(i32 out, i32 in, f32 w) -> void {
        weights_[out * inCount_ + in] = w;
    }

    auto weight(i32 out, i32 in) -> f32 {
        return weights_[out * inCount_ + in];
    }

    auto outCount() const -> i32 { return outCount_; }

    auto inCount() const -> i32 { return inCount_; }

private:
    std::vector<f32> weights_;
    i32 inCount_;
    i32 outCount_;
};


class Layer {
public:
    Layer(size_t neuronCount) : neurons_(neuronCount) {}

    auto getNeuron(i32 index) -> Neuron & { return neurons_[index]; }

    auto size() const -> size_t { return neurons_.size(); }

private:
    std::vector<Neuron> neurons_;
};

class NeuralNetwork {
public:
    explicit NeuralNetwork(const size_t *layersSizeList, size_t layersCount)  {
        for (int i = 0; i < layersCount; i++) {
            if (i != 0)
                weightMatrix_.emplace_back(layersSizeList[i], layersSizeList[i - 1]);
            layers_.emplace_back(layersSizeList[i]);
        }
    }

public:
    auto getResult(i32 i) -> f32 {
        return layers_[layers_.size() - 1].getNeuron(i).value;
    }

    auto predict(const f32 *inputData) {
        auto &inputLayer = layers_[0];

        for (int i = 0; i < inputLayer.size(); i++) {
            inputLayer.getNeuron(i).value = inputData[i];
        }

        for (int k = 1; k < layers_.size(); k++) {
            auto& layer = layers_[k];
            auto& weightsBetween = weightMatrix_[k - 1];
            auto& prevLayer = layers_[k - 1];

            for (int i = 0; i < layer.size(); i++) {
                auto &neuron = layer.getNeuron(i);

                auto sum = 0.f;
                for (int j = 0; j < prevLayer.size(); j++) {
                    auto w = weightsBetween.weight(i, j);
                    auto pn = prevLayer.getNeuron(j).value;

                    sum += w * pn;
                }

                neuron.value = activate(sum);
            }
        }
    }

    auto train(size_t trainDataCount, const f32 *inputData, const f32 *outputData) -> void {
        for (i32 epoch = 0; epoch < 200'000; epoch++) {
            auto isError = false;

            for (int i = 0; i < trainDataCount; i++) {
                predict(inputData + i * layers_[0].size());

                auto expected = outputData + i * layers_[layers_.size() - 1].size();

                auto err = calculateError(expected);

                if (err > 0.025f) {

                    adjust(expected);
                    predict(inputData + i * layers_[0].size());
                    isError = true;
                }
            }

            if (!isError) break;
        }
    }

    auto calculateError(const f32 *expectedOutputList) -> f32 {
        auto sum = 0.f;

        auto &endLayer = layers_[layers_.size() - 1];

        for (int i = 0; i < endLayer.size(); i++) {
            sum += abs(expectedOutputList[i] - endLayer.getNeuron(i).value);
        }

        return sum / 2.f;
    }

    auto adjust(const f32 *&expectedOutputList, f32 alpha = 0.1f) -> void {
        for (int k = layers_.size() - 1; k >= 1; k--) {
            auto &weights = weightMatrix_[k - 1];

            for (int i = 0; i < weights.outCount(); i++) {
                auto beta = computeBeta(k, i, expectedOutputList);

                auto &neuron = layers_[k].getNeuron(i);

                neuron.beta = beta;

                for (int j = 0; j < weights.inCount(); j++) {
                    auto delta = alpha * beta * layers_[k - 1].getNeuron(j).value;

                    auto newWeight = weights.weight(i, j) + delta;

                    weights.setWeight(i, j, newWeight);
                }
            }
        }
    }

private:
    auto computeBeta(i32 layerIndex, i32 neuronIndex, const f32 *expectedOutputList) -> f32 {
        auto &neuron = layers_[layerIndex].getNeuron(neuronIndex);

        auto y = neuron.value;
        auto beta = y * (1.f - y);

        if (layerIndex == layers_.size() - 1) {
            beta *= expectedOutputList[neuronIndex] - y;
        } else {
            auto &nextLayer = layers_[layerIndex + 1];
            auto &nextWeights = weightMatrix_[layerIndex];

            auto sum = 0.f;
            for (int j = 0; j < nextLayer.size(); j++) {
                sum += nextLayer.getNeuron(j).beta * nextWeights.weight(j, neuronIndex);
            }
            beta *= sum;
        }

        return beta;
    }


private:
    std::vector<Layer> layers_;
    std::vector<WeightMatrix> weightMatrix_;
};

auto test() -> void {
    size_t layers[] = {2, 2 };
    NeuralNetwork nn(layers, sizeof(layers) / sizeof(layers[0]));

    nn.predict(new f32[]{1.f, 0.25f});

    //               0.4200093864
    //  - ( 1.00f ) ---- (   )
    //  0.391549617 \  /
    //               \/
    //               /\
    //  0.197191462 /  \
    //  - ( 0.25f )  ---- (  )
    //              0.39922002
}

auto main() -> int {
    size_t layers[] = {2, 4, 1};

    NeuralNetwork nn(layers, sizeof(layers) / sizeof(layers[0]));

    f32 trainDataIn[] = {
            0.f, 0.f,
            0.f, 1.f,
            1.f, 0.f,
            1.f, 1.f
    };

    f32 trainDataOut[] = {
            0, 1.f, 1.f, 0.f
    };

    nn.train(4, trainDataIn, trainDataOut);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float input[] = {(f32) i + 0.5f, (f32) j  + 0.5f};
            nn.predict(input);
            std::cout << i << " xor " << j << " = " << nn.getResult(0) << std::endl;
        }
    }

    return 0;
}

