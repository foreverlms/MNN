#include "lms_source.hpp"
#include <MNN/Interpreter.hpp>
#include <fstream>
#include <iostream>

namespace lms {

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

std::shared_ptr<char> ReadBinaryFile(const std::string &filePath, size_t &size) {
  std::ifstream file(filePath, std::ios::in | std::ios::binary);
  if (!file.good()) {
    return nullptr;
  }
  if (!file.is_open()) {
    return nullptr;
  }

  file.seekg(0, std::ios::end);
  size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::shared_ptr<char> buffer = std::shared_ptr<char>(new char[size], std::default_delete<char[]>());
  if (!buffer) {
    return nullptr;
  }
  file.read(buffer.get(), size);
  file.close();
  return buffer;
}

bool test_specified_model(const std::string &modelPath) {
  auto net = MNN::Interpreter::createFromFile(modelPath.c_str());
  if (NULL == net) {
    return false;
  }
  MNN::ScheduleConfig cpuconfig;
  cpuconfig.type = MNN_FORWARD_CPU;
  MNN::BackendConfig backendConfig;
  backendConfig.memory = MNN::BackendConfig::Memory_High;
  cpuconfig.backendConfig = &backendConfig;
  auto CPU = net->createSession(cpuconfig);

  // multiple inputs
  for (const auto &pair : net->getSessionInputAll(CPU)) {
    auto &input = pair.second;
    auto nhwcTensor = new MNN::Tensor(input, MNN::Tensor::TENSORFLOW);
    for (int i = 0; i < nhwcTensor->size(); ++i) {
      nhwcTensor->host<float>()[i] = 1.0f;
    }
    input->copyFromHostTensor(nhwcTensor);
    delete nhwcTensor;
  }

  PROGRAM_BEGIN
    net->runSession(CPU);
  PROGRAM_END

  for (const auto &pair : net->getSessionOutputAll(CPU)) {
    std::cout << "Get Output Name: " << pair.first << std::endl;
    const auto &output = pair.second;
    auto outputTensor = new MNN::Tensor(output, MNN::Tensor::CAFFE); // lms: 这里对ONNX和PB可不一样
    output->copyToHostTensor(outputTensor);
    const auto *ptr = outputTensor->host<float>();
//    for (int i = 0; i < outputTensor->size(); ++i) {
//      std::cout << "Index: " << i << ",Computed: " << ptr[i] << std::endl;
//    }
    delete outputTensor;
  }
}

bool test_fixed_model() {
  std::string modelPath = "/Users/bob/code/CodeReading/mnn/MNN/cmake-build-debug/conv1x1.mnn";
  std::string inputPath = "/Users/bob/code/CodeReading/mnn/MNN/cmake-build-debug/input.bin";
  std::string outputPath = "/Users/bob/code/CodeReading/mnn/MNN/cmake-build-debug/output.bin";

  size_t bytesOfInput, bytesOfOutput;
  const auto *inputData = std::reinterpret_pointer_cast<float>(ReadBinaryFile(inputPath, bytesOfInput)).get();
  auto realSmartData = ReadBinaryFile(outputPath, bytesOfOutput);
  const auto *realData = (float *) realSmartData.get();

  assert(inputData);
  assert(realData);

  auto net = MNN::Interpreter::createFromFile(modelPath.c_str());
  if (NULL == net) {
    return false;
  }
  MNN::ScheduleConfig cpuconfig;
  cpuconfig.type = MNN_FORWARD_CPU;
  MNN::BackendConfig backendConfig;
  backendConfig.memory = MNN::BackendConfig::Memory_High;
  cpuconfig.backendConfig = &backendConfig;
  auto CPU = net->createSession(cpuconfig);

  // multiple inputs
  //    for (const auto &pair : net->getSessionInputAll(CPU))
  //    {
  //        auto &input = pair.second;
  //        auto nhwcTensor = new MNN::Tensor(input, MNN::Tensor::TENSORFLOW);
  //        for (int i = 0; i < nhwcTensor->size(); ++i)
  //        {
  //            nhwcTensor->host<float>()[i] = 1.0f;
  //        }
  //        input->copyFromHostTensor(nhwcTensor);
  //        delete nhwcTensor;
  //    }

  auto input = net->getSessionInput(CPU, NULL);

  auto nchwTensor = new MNN::Tensor(input, MNN::Tensor::CAFFE);
  for (int i = 0; i < nchwTensor->size(); ++i) {
    nchwTensor->host<float>()[i] = inputData[i];
  }

  input->copyFromHostTensor(nchwTensor);
  delete nchwTensor;

  PROGRAM_BEGIN
    net->runSession(CPU);
  PROGRAM_END

  auto output = net->getSessionOutput(CPU, "output");
  auto outputTensor = new MNN::Tensor(output, MNN::Tensor::CAFFE);
  output->copyToHostTensor(outputTensor);

  float tolerance = 0.08;

//  const auto *ptr = outputTensor->host<float>();
//  for (int i = 0; i < outputTensor->size(); ++i) {
//    auto diff = std::abs(ptr[i] - realData[i]) / realData[i];
//    if (diff > tolerance) {
//      std::cout << "Index: " << i << ",Computed: " << ptr[i] << ", Real: " << realData[i] << std::endl;
//      assert(false);
//    }
//  }
  delete outputTensor;
}

} // namespace lms
