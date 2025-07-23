#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>

class ORTModel {
public:
    explicit ORTModel(const std::string &model_path)
            : env_(ORT_LOGGING_LEVEL_WARNING, "pc_test"),
              sess_opts_{},
              session_{nullptr} {
        sess_opts_.SetIntraOpNumThreads(1);
        sess_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = Ort::Session(env_, model_path.c_str(), sess_opts_);

        /* IO 名字 */
        Ort::AllocatorWithDefaultOptions alloc;
        input_name_ = session_.GetInputNameAllocated(0, alloc).get();
        output_name_ = session_.GetOutputNameAllocated(0, alloc).get();

        /* 输入 shape */
        auto inTypeInfo = session_.GetInputTypeInfo(0);
        auto inTensorInfo = inTypeInfo.GetTensorTypeAndShapeInfo();
        input_shape_ = inTensorInfo.GetShape();
        input_size_ = 1;
        for (auto s: input_shape_) { if (s > 0) input_size_ *= s; }
    }

    /* run: 传入 float* 指针, 返回 sigmoid(score) */
    float run(const float *data) {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, const_cast<float *>(data),
                input_size_, input_shape_.data(), input_shape_.size());

        std::array<const char *, 1> in_names = {input_name_.c_str()};
        std::array<const char *, 1> out_names = {output_name_.c_str()};

        auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
                                           in_names.data(), &input_tensor, 1,
                                           out_names.data(), 1);

        float *out = output_tensors[0].GetTensorMutableData<float>();
        float raw = out[0];
        return 1.f / (1.f + std::exp(-raw));          // sigmoid
    }

    size_t input_size() const { return input_size_; }

private:
    Ort::Env env_;
    Ort::SessionOptions sess_opts_;
    Ort::Session session_;
    std::string input_name_, output_name_;
    std::vector<int64_t> input_shape_;
    size_t input_size_{};
};