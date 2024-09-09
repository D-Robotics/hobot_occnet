#include "hobot_occnet/occnet_node.h"
#include <cassert>

OccNetNode::OccNetNode(const std::string &node_name, const rclcpp::NodeOptions &node_options) : Node(node_name, node_options)
{
    RCLCPP_INFO(this->get_logger(), "=> init %s", node_name.c_str());
    // =================================================================================================================================
    this->declare_parameter("occ_model_file_path", "");
    this->declare_parameter("local_image_dir", "");
    this->get_parameter("occ_model_file_path", occ_model_file_path_);
    this->get_parameter("local_image_dir", local_image_dir_);
    RCLCPP_INFO_STREAM(this->get_logger(), "=> occ_model_file_path: " << occ_model_file_path_);
    RCLCPP_INFO_STREAM(this->get_logger(), "=> local_image_dir: " << local_image_dir_);

    infer_test();
}

void OccNetNode::infer_callback(const sensor_msgs::msg::Image::ConstSharedPtr &stereo_msg)
{
}

void OccNetNode::infer_test()
{
    RCLCPP_INFO(this->get_logger(), "=>==================== Load model");
    // 第一步加载模型
    hbPackedDNNHandle_t packed_dnn_handle;
    const char *occ_model_file_path_cstr = occ_model_file_path_.c_str();
    hbDNNInitializeFromFiles(&packed_dnn_handle, &occ_model_file_path_cstr, 1);

    RCLCPP_INFO(this->get_logger(), "=>==================== Get Model Name");
    // 第二步获取模型名称
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    for (int i = 0; i < model_count; i++)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "=> model " << i << " : " << model_name_list[i]);
    }

    RCLCPP_INFO(this->get_logger(), "=>==================== Get Dnn Handle");
    // 第三步获取dnn_handle
    hbDNNHandle_t dnn_handle;
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    RCLCPP_INFO(this->get_logger(), "=>==================== Prepare Data Mem");
    // 第四步准备输入、输出数据空间
    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;
    prepare_input_tensor(input_tensors, dnn_handle);
    prepare_output_tensor(output_tensors, dnn_handle);
    int input_count = 0;
    int output_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    hbDNNGetOutputCount(&output_count, dnn_handle);
    RCLCPP_INFO_STREAM(this->get_logger(), "=> input_count: " << input_count << " , output_count: " << output_count);
    // input_tensors.resize(input_count);
    // output_tensors.resize(output_count);
    // std::vector的data()成员函数返回一个指向向量内存中存储的第一个元素的指针
    // prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle);

    RCLCPP_INFO(this->get_logger(), "=>==================== Set Input Data");
    // 第五步读入图像到输入数据
    for (const auto &entry : fs::directory_iterator(local_image_dir_))
    {
        std::string filename = entry.path().filename().string();
        if (filename.find("Left") == 0)
        {
            std::string left_img_path = entry.path().string();
            std::string right_img_path = entry.path().parent_path().string() + "/" + "Right" + filename.substr(4);
            RCLCPP_INFO_STREAM(this->get_logger(), "=> left_img_path: " << left_img_path << " , right_img_path: " << right_img_path);
            cv::Mat left_img_bgr = cv::imread(left_img_path, cv::IMREAD_COLOR);
            cv::Mat right_img_bgr = cv::imread(right_img_path, cv::IMREAD_COLOR);
            if (left_img_bgr.empty() || right_img_bgr.empty())
            {
                RCLCPP_ERROR(this->get_logger(), "=> read imgs fail!");
            }

            cv::Mat left_img_nv12, right_img_nv12;
            utils::bgr_to_nv12_mat(left_img_bgr, left_img_nv12);
            utils::bgr_to_nv12_mat(right_img_bgr, right_img_nv12);
            int left_nv12_memsize = left_img_nv12.rows * left_img_nv12.cols * left_img_nv12.channels() * left_img_nv12.elemSize();
            int right_nv12_memsize = right_img_nv12.rows * right_img_nv12.cols * right_img_nv12.channels() * right_img_nv12.elemSize();

            RCLCPP_INFO_STREAM(this->get_logger(), "=> left_nv12_memsize: " << left_nv12_memsize << ", right_nv12_memsize: " << right_nv12_memsize
                                                                            << ", total: " << left_nv12_memsize + right_nv12_memsize);
            hbDNNTensor &left_input_tensor = input_tensors[0], &right_input_tensor = input_tensors[1];
            hbSysWriteMem(&left_input_tensor.sysMem[0], (char *)left_img_nv12.data, left_input_tensor.sysMem[0].memSize);
            hbSysWriteMem(&left_input_tensor.sysMem[1], (char *)left_img_nv12.data + left_input_tensor.sysMem[0].memSize,
                          left_input_tensor.sysMem[1].memSize);

            hbSysWriteMem(&right_input_tensor.sysMem[0], (char *)right_img_nv12.data, right_input_tensor.sysMem[0].memSize);
            hbSysWriteMem(&right_input_tensor.sysMem[1], (char *)right_img_nv12.data + right_input_tensor.sysMem[0].memSize,
                          right_input_tensor.sysMem[1].memSize);

            // make sure memory data is flushed to DDR before inference
            hbSysFlushMem(&left_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
            hbSysFlushMem(&left_input_tensor.sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
            hbSysFlushMem(&right_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
            hbSysFlushMem(&right_input_tensor.sysMem[1], HB_SYS_MEM_CACHE_CLEAN);

            RCLCPP_INFO(this->get_logger(), "=>==================== Run Infer");
            // 第六步推理模型
            hbDNNTensor *output = output_tensors.data();
            hbDNNInferCtrlParam infer_ctrl_param;
            HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
            // 耗时测试
            auto before_infer = std::chrono::system_clock::now();
            hbDNNTaskHandle_t task_handle = nullptr;
            hbDNNInfer(&task_handle, &output, input_tensors.data(), dnn_handle, &infer_ctrl_param);
            // wait task done
            hbDNNWaitTaskDone(task_handle, 0);
            hbDNNReleaseTask(task_handle);
            auto after_infer = std::chrono::system_clock::now();
            auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(after_infer - before_infer).count();
            RCLCPP_INFO_STREAM(this->get_logger(), "time cost: " << interval << " ms, fps: " << 1 / (interval / 1000.0));

            RCLCPP_INFO(this->get_logger(), "=>==================== Process Output");
            // 第七步处理结果
            // make sure CPU read data from DDR before using output tensor data
            for (int i = 0; i < output_count; i++)
            {
                hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
            }
            postprocess(output_tensors, filename + ".txt");
        }
    }
}

int OccNetNode::prepare_input_tensor(std::vector<hbDNNTensor> &input_tensors, hbDNNHandle_t dnn_handle)
{
    int model_h, model_w;
    input_tensors.resize(2);

    hbDNNTensorProperties properties;
    for (auto &tensor : input_tensors)
    {
        hbDNNGetInputTensorProperties(&properties, dnn_handle, 0);
        tensor.properties = properties;
        tensor.properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE;
        switch (properties.tensorLayout)
        {
        case HB_DNN_LAYOUT_NHWC:
            model_h = properties.validShape.dimensionSize[1];
            model_w = properties.validShape.dimensionSize[2];
            break;
        case HB_DNN_LAYOUT_NCHW:
            model_h = properties.validShape.dimensionSize[2];
            model_w = properties.validShape.dimensionSize[3];
            break;
        default:
            return -1;
        }
        RCLCPP_INFO_STREAM(this->get_logger(), "=> model_h: " << model_h << ", model_w: " << model_w);
        tensor.properties.validShape.numDimensions = 4;
        tensor.properties.validShape.dimensionSize[0] = 1;
        tensor.properties.validShape.dimensionSize[1] = 3;
        tensor.properties.validShape.dimensionSize[2] = model_h;
        tensor.properties.validShape.dimensionSize[3] = model_w;
        tensor.properties.alignedShape = tensor.properties.validShape;

        hbSysAllocCachedMem(&tensor.sysMem[0], model_h * model_w);
        tensor.sysMem[0].memSize = model_h * model_w;

        hbSysAllocCachedMem(&tensor.sysMem[1], model_h * model_w / 2);
        tensor.sysMem[1].memSize = model_h * model_w / 2;
    }
    return 0;
}

int OccNetNode::prepare_output_tensor(std::vector<hbDNNTensor> &output_tensors, hbDNNHandle_t dnn_handle)
{
    int output_count = 0;
    hbDNNGetOutputCount(&output_count, dnn_handle);
    output_tensors.resize(output_count);
    for (int i = 0; i < output_count; ++i)
    {
        hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i);
        RCLCPP_INFO_STREAM(this->get_logger(), "=> output[" << i << "].memsize: " << output_tensors[i].properties.alignedByteSize);
        hbSysAllocCachedMem(&output_tensors[i].sysMem[0], output_tensors[i].properties.alignedByteSize);
    }
    return 0;
}

int OccNetNode::postprocess(std::vector<hbDNNTensor> &output_tensors, std::string filename)
{
    hbDNNTensor output_tensor = output_tensors[1];
    assert(output_tensor.properties.tensorType == HB_DNN_TENSOR_TYPE_S16);

    //
    auto output_tensor_addr = reinterpret_cast<uint16_t *>(output_tensor.sysMem[0].virAddr);
    auto shift = output_tensor.properties.shift.shiftData;
    auto scale = output_tensor.properties.scale.scaleData;

    //
    int *shape = output_tensor.properties.validShape.dimensionSize;
    int B = shape[0], C = shape[1], H = shape[2], W = shape[3];
    RCLCPP_INFO(this->get_logger(), "=> shape: [%d, %d, %d, %d]", B, C, H, W);
    std::ofstream outFile(filename);
    int count = 0;
    for (int b = 0; b < B; b++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                for (int c = 0; c < C; c++)
                {
                    int index = b * C * H * W + c * H * W + h * W + w;
                    uint16_t occ_val = output_tensor_addr[index];
                    if (count < 100)
                    {
                        RCLCPP_INFO_STREAM(this->get_logger(), index << ", " << occ_val << ", " << scale[index] << "; ");
                        count++;
                    }
                    outFile << occ_val; // 写入元素
                    if (c < C - 1)
                    { // 如果不是最后一个元素，添加逗号
                        outFile << ",";
                    }
                }
                outFile << "\n";
            }
        }
    }
    outFile.close();
    RCLCPP_INFO(this->get_logger(), "=> done!");
}