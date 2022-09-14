#include <MNN/Interpreter.hpp>
#include <fstream>
#include <iostream>

namespace lms
{

#define PROGRAM_BEGIN \
    {                 \
        auto __cppdemo_start__ = std::chrono::high_resolution_clock::now();

#define PROGRAM_END                                                                                                             \
    auto __cppdemo_end__ = std::chrono::high_resolution_clock::now();                                                           \
    auto __cppdemo_microseconds__ = std::chrono::duration_cast<std::chrono::microseconds>(__cppdemo_end__ - __cppdemo_start__); \
    auto __cpdemo_milliseconds__ = std::chrono::duration_cast<std::chrono::milliseconds>(__cppdemo_microseconds__);             \
    std::cout << "Time Elapsed:\t" << __cppdemo_microseconds__.count() << " us"                                                 \
              << ", namely: " << __cpdemo_milliseconds__.count() << " ms" << std::endl;                                         \
    }

std::shared_ptr<char> ReadBinaryFile(const std::string &filePath, size_t &size)
{
    std::ifstream file(filePath, std::ios::in | std::ios::binary);
    if (!file.good())
    {
        return nullptr;
    }
    if (!file.is_open())
    {
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::shared_ptr<char> buffer = std::shared_ptr<char>(new char[size], std::default_delete<char[]>());
    if (!buffer)
    {
        return nullptr;
    }
    file.read(buffer.get(), size);
    file.close();
    return buffer;
}

bool test_quality()
{
    std::string modelPath = "/Users/bob/docs/ByteDance/PNN/converter/deepfm/trival_deep_fm.mnn";
    std::string inputPath = "/Users/bob/docs/ByteDance/PNN/converter/quality/onnx_input.bin";
    std::string outputPath = "/Users/bob/docs/ByteDance/PNN/converter/quality/onnx_output.bin";
    std::string upsampleOutputPath = "/Users/bob/docs/ByteDance/PNN/converter/quality/upsample_output.bin";

    size_t bytesOfInput, bytesOfOutput;
    auto *inputData = std::reinterpret_pointer_cast<float>(ReadBinaryFile(inputPath, bytesOfInput)).get();
    auto realSmartData = ReadBinaryFile(outputPath, bytesOfOutput);
    auto *realData = (float *) realSmartData.get();
    auto upsampleSmartData = ReadBinaryFile(upsampleOutputPath, bytesOfOutput);
    auto *upsampleData = (float *) upsampleSmartData.get();

    assert(inputData);
    assert(realData);

    auto net = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (NULL == net)
    {
        return false;
    }
    MNN::ScheduleConfig cpuconfig;
    cpuconfig.type = MNN_FORWARD_CPU;
    auto CPU = net->createSession(cpuconfig);

    // multiple inputs
    for (const auto &pair : net->getSessionInputAll(CPU))
    {
        auto &input = pair.second;
        auto nhwcTensor = new MNN::Tensor(input, MNN::Tensor::TENSORFLOW);
        for (int i = 0; i < nhwcTensor->size(); ++i)
        {
            nhwcTensor->host<float>()[i] = 1.0f;
        }
        input->copyFromHostTensor(nhwcTensor);
        delete nhwcTensor;
    }

    //    auto input = net->getSessionInput(CPU, NULL);
    //    auto nchwTensor = new MNN::Tensor(input, MNN::Tensor::CAFFE);
    //    for (int i = 0; i < nchwTensor->size(); ++i)
    //    {
    //        nchwTensor->host<float>()[i] = inputData[i];
    //    }

    //    input->copyFromHostTensor(nchwTensor);
    //    delete nchwTensor;

    PROGRAM_BEGIN
    net->runSession(CPU);
    PROGRAM_END

    auto output = net->getSessionOutput(CPU, "prediction");
    auto outputTensor = new MNN::Tensor(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(outputTensor);

//    auto upsample = net->getSessionOutput(CPU, "644");
//    auto upsampleOutput = new MNN::Tensor(upsample, MNN::Tensor::CAFFE);
//    upsample->copyToHostTensor(upsampleOutput);

    float tolerance = 0.08;

    //    const auto *upsamplePtr = upsampleOutput->host<float>();
    //    for (int i = 0; i < upsampleOutput->size(); ++i)
    //    {
    //
    //        auto diff = std::abs(upsamplePtr[i] - upsampleData[i]) / upsampleData[i];
    //        if (diff > tolerance)
    //        {
    //            std::cout << "Upsample Index: " << i << ",Computed: " << upsamplePtr[i] << ", Real: " << upsampleData[i] << std::endl;
    //            assert(false);
    //        }
    //    }
    //    delete upsampleOutput;

    const auto *ptr = outputTensor->host<float>();
    for (int i = 0; i < outputTensor->size(); ++i)
    {
        auto diff = std::abs(ptr[i] - realData[i]) / realData[i];
        if (diff > tolerance)
        {
            std::cout << "Index: " << i << ",Computed: " << ptr[i] << ", Real: " << realData[i] << std::endl;
            assert(false);
        }
    }
    delete outputTensor;
}
} // namespace lms

int main(void)
{
    lms::test_quality();
}
