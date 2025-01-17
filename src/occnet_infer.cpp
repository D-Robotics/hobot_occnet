#include "hobot_occnet/occnet_infer.h"

int OccNetInfer::init(std::string &occ_model_file_path)
{
    int ret_code = 0;
    RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> ==================== init occ model start ====================");
    // load model
    occ_model_file_path_ = occ_model_file_path;
    const char *occ_model_file_path_cstr = occ_model_file_path_.c_str();
    ret_code = hbDNNInitializeFromFiles(&packed_dnn_handle_, &occ_model_file_path_cstr, 1);
    HB_CHECK_SUCCESS(ret_code, "hbDNNInitializeFromFiles failed");

    // get model name
    ret_code = hbDNNGetModelNameList(&model_name_list_, &model_count_, packed_dnn_handle_);
    HB_CHECK_SUCCESS(ret_code, "hbDNNGetModelNameList failed");

    // get model handle
    ret_code = hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list_[0]);
    HB_CHECK_SUCCESS(ret_code, "hbDNNGetModelHandle failed");

    // get input count and output count
    ret_code = hbDNNGetInputCount(&input_count_, dnn_handle_);
    HB_CHECK_SUCCESS(ret_code, "hbDNNGetInputCount failed");
    ret_code = hbDNNGetOutputCount(&output_count_, dnn_handle_);
    HB_CHECK_SUCCESS(ret_code, "hbDNNGetOutputCount failed");
    RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"), "=> model name: " << model_name_list_[0]);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"), "=> input_count: " << input_count_);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"), "=> output_count: " << output_count_);

    // allocate memory for input/output tensor
    ret_code = prepare_input_tensor_nv12();
    HB_CHECK_SUCCESS(ret_code, "prepare_input_tensor_nv12 failed");
    ret_code = prepare_output_tensor();
    HB_CHECK_SUCCESS(ret_code, "prepare_output_tensor failed");

    RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> ==================== init occ model end   ====================");

    return ret_code;
}

int OccNetInfer::prepare_input_tensor_nv12()
{
    int ret_code = 0;
    RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> ----- prepare_input_tensor_nv12 -----");
    // check the type of input tensor
    hbDNNTensorProperties properties;
    ret_code = hbDNNGetInputTensorProperties(&properties, dnn_handle_, 0);
    HB_CHECK_SUCCESS(ret_code, "hbDNNGetInputTensorProperties failed");
    RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"), "=> input tensor type is " << tensor_type_to_str(properties.tensorType));
    if ((properties.tensorType != HB_DNN_IMG_TYPE_NV12) && (properties.tensorType != HB_DNN_IMG_TYPE_NV12_SEPARATE))
    {
        RCLCPP_ERROR(rclcpp::get_logger("OccnetInfer"), "=> input tensor type is not in [HB_DNN_IMG_TYPE_NV12, HB_DNN_IMG_TYPE_NV12_SEPARATE]");
        return -1;
    }
    RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"), "=> input tensor memsize: " << properties.alignedByteSize);
    input_tensor_type_ = properties.tensorType;
    // int dims = properties.validShape.numDimensions;
    // int *shape = properties.validShape.dimensionSize;
    // RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> input tensor dims: %d", dims);
    // RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> input tensor shape: [%d, %d, %d, %d]", shape[0], shape[1], shape[2], shape[3]);

    // allocate memory for input tensor
    input_tensors_.resize(2);
    for (auto &tensor : input_tensors_)
    {
        tensor.properties = properties;
        tensor.properties.tensorType = properties.tensorType;
        switch (properties.tensorLayout)
        {
        case HB_DNN_LAYOUT_NHWC:
            model_input_h_ = properties.validShape.dimensionSize[1];
            model_input_w_ = properties.validShape.dimensionSize[2];
            break;
        case HB_DNN_LAYOUT_NCHW:
            model_input_h_ = properties.validShape.dimensionSize[2];
            model_input_w_ = properties.validShape.dimensionSize[3];
            break;
        default:
            RCLCPP_ERROR(rclcpp::get_logger("OccnetInfer"), "=> input tensor layout is not in [HB_DNN_LAYOUT_NHWC, HB_DNN_LAYOUT_NCHW]");
            return -1;
        }
        tensor.properties.validShape.numDimensions = 4;
        tensor.properties.validShape.dimensionSize[0] = 1;
        tensor.properties.validShape.dimensionSize[1] = 3;
        tensor.properties.validShape.dimensionSize[2] = model_input_h_;
        tensor.properties.validShape.dimensionSize[3] = model_input_w_;
        tensor.properties.alignedShape = tensor.properties.validShape;

        RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"), "=> model_input_h: " << model_input_h_ << ", model_input_w: " << model_input_w_);

        if (properties.tensorType == HB_DNN_IMG_TYPE_NV12)
        {
            RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> allocate memory HB_DNN_IMG_TYPE_NV12");
            ret_code = hbSysAllocCachedMem(&tensor.sysMem[0], (3 * model_input_h_ * model_input_w_) / 2);
            HB_CHECK_SUCCESS(ret_code, "hbSysAllocCachedMem failed");
            tensor.sysMem[0].memSize = (3 * model_input_h_ * model_input_w_) / 2;
        }
        else if (properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE)
        {
            RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> allocate memory HB_DNN_IMG_TYPE_NV12_SEPARATE");
            ret_code = hbSysAllocCachedMem(&tensor.sysMem[0], model_input_h_ * model_input_w_);
            HB_CHECK_SUCCESS(ret_code, "hbSysAllocCachedMem failed");
            tensor.sysMem[0].memSize = model_input_h_ * model_input_w_;

            ret_code = hbSysAllocCachedMem(&tensor.sysMem[1], model_input_h_ * model_input_w_ / 2);
            HB_CHECK_SUCCESS(ret_code, "hbSysAllocCachedMem failed");
            tensor.sysMem[1].memSize = model_input_h_ * model_input_w_ / 2;
        }
        else
        {
            return -1;
        }
    }
    return ret_code;
}

int OccNetInfer::prepare_output_tensor()
{
    int ret_code = 0;
    RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> ----- prepare_output_tensor -----");
    output_tensors_.resize(output_count_);
    for (int i = 0; i < output_count_; ++i)
    {
        ret_code = hbDNNGetOutputTensorProperties(&output_tensors_[i].properties, dnn_handle_, i);
        HB_CHECK_SUCCESS(ret_code, "hbDNNGetOutputTensorProperties failed");
        ret_code = hbSysAllocCachedMem(&output_tensors_[i].sysMem[0], output_tensors_[i].properties.alignedByteSize);
        HB_CHECK_SUCCESS(ret_code, "hbSysAllocCachedMem failed");
        RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"), "=> output[" << i << "].memsize: " << output_tensors_[i].properties.alignedByteSize);
    }
    return ret_code;
}

int OccNetInfer::fill_bgr_to_tensor_nv12(const cv::Mat &left_img_bgr, const cv::Mat &right_img_bgr)
{
    int ret_code = 0;
    RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> ----- fill_bgr_to_tensor_nv12 -----");
    cv::Mat left_img_nv12, right_img_nv12;
    utils::bgr_to_nv12_mat(left_img_bgr, left_img_nv12);
    utils::bgr_to_nv12_mat(right_img_bgr, right_img_nv12);
    int left_nv12_memsize = left_img_nv12.rows * left_img_nv12.cols * left_img_nv12.channels() * left_img_nv12.elemSize();
    int right_nv12_memsize = right_img_nv12.rows * right_img_nv12.cols * right_img_nv12.channels() * right_img_nv12.elemSize();
    RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"),
                       "=> left_nv12_memsize: " << left_nv12_memsize << ", right_nv12_memsize: " << right_nv12_memsize << ", total: " << left_nv12_memsize + right_nv12_memsize);

    hbDNNTensor &left_input_tensor = input_tensors_[0];
    hbDNNTensor &right_input_tensor = input_tensors_[1];

    if (input_tensor_type_ == HB_DNN_IMG_TYPE_NV12)
    {
        RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> fill image data into memory HB_DNN_IMG_TYPE_NV12");
        // fill image data into memory
        ret_code = hbSysWriteMem(&left_input_tensor.sysMem[0], (char *)left_img_nv12.data, left_input_tensor.sysMem[0].memSize);
        HB_CHECK_SUCCESS(ret_code, "hbSysWriteMem failed");
        ret_code = hbSysWriteMem(&right_input_tensor.sysMem[0], (char *)right_img_nv12.data, right_input_tensor.sysMem[0].memSize);
        HB_CHECK_SUCCESS(ret_code, "hbSysWriteMem failed");
        // make sure memory data is flushed to DDR before inference
        ret_code = hbSysFlushMem(&left_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
        HB_CHECK_SUCCESS(ret_code, "hbSysFlushMem failed");
        ret_code = hbSysFlushMem(&right_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
        HB_CHECK_SUCCESS(ret_code, "hbSysFlushMem failed");
    }
    else if (input_tensor_type_ == HB_DNN_IMG_TYPE_NV12_SEPARATE)
    {
        RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=>fill image data into memory HB_DNN_IMG_TYPE_NV12_SEPARATE");
        // fill image data into memory
        ret_code = hbSysWriteMem(&left_input_tensor.sysMem[0], (char *)left_img_nv12.data, left_input_tensor.sysMem[0].memSize);
        HB_CHECK_SUCCESS(ret_code, "hbSysWriteMem failed");
        ret_code = hbSysWriteMem(&left_input_tensor.sysMem[1], (char *)left_img_nv12.data + left_input_tensor.sysMem[0].memSize, left_input_tensor.sysMem[1].memSize);
        HB_CHECK_SUCCESS(ret_code, "hbSysWriteMem failed");
        ret_code = hbSysWriteMem(&right_input_tensor.sysMem[0], (char *)right_img_nv12.data, right_input_tensor.sysMem[0].memSize);
        HB_CHECK_SUCCESS(ret_code, "hbSysWriteMem failed");
        ret_code = hbSysWriteMem(&right_input_tensor.sysMem[1], (char *)right_img_nv12.data + right_input_tensor.sysMem[0].memSize, right_input_tensor.sysMem[1].memSize);
        HB_CHECK_SUCCESS(ret_code, "hbSysWriteMem failed");

        // make sure memory data is flushed to DDR before inference
        ret_code = hbSysFlushMem(&left_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
        HB_CHECK_SUCCESS(ret_code, "hbSysFlushMem failed");
        ret_code = hbSysFlushMem(&left_input_tensor.sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
        HB_CHECK_SUCCESS(ret_code, "hbSysFlushMem failed");
        ret_code = hbSysFlushMem(&right_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
        HB_CHECK_SUCCESS(ret_code, "hbSysFlushMem failed");
        ret_code = hbSysFlushMem(&right_input_tensor.sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
        HB_CHECK_SUCCESS(ret_code, "hbSysFlushMem failed");
    }
    else
    {
        RCLCPP_ERROR(rclcpp::get_logger("OccnetInfer"), "=> input_tensor_type is not in [HB_DNN_IMG_TYPE_NV12, HB_DNN_IMG_TYPE_NV12_SEPARATE]");
        return -1;
    }

    return ret_code;
}

int OccNetInfer::forward(const cv::Mat &left_img, const cv::Mat &right_img, const InputImgType &input_img_type)
{
    int ret_code = 0;
    RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"), "=> ==================== infer by model =======================");
    if (input_img_type == InputImgType::BGR8)
    {
        RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> fill bgr8 image to tensor");
        fill_bgr_to_tensor_nv12(left_img, right_img);
    }
    else if (input_img_type == InputImgType::NV12)
    {
        RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> fill nv12 image to tensor");
    }
    else
    {
        return -1;
    }
    RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> ----- infer -----");
    hbDNNTensor *output = output_tensors_.data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    // == time-consuming test ==
    auto before_infer = std::chrono::system_clock::now();
    hbDNNTaskHandle_t task_handle = nullptr;
    ret_code = hbDNNInfer(&task_handle, &output, input_tensors_.data(), dnn_handle_, &infer_ctrl_param);
    HB_CHECK_SUCCESS(ret_code, "hbDNNInfer failed");
    // wait task done
    ret_code = hbDNNWaitTaskDone(task_handle, 0);
    HB_CHECK_SUCCESS(ret_code, "hbDNNWaitTaskDone failed");
    ret_code = hbDNNReleaseTask(task_handle);
    HB_CHECK_SUCCESS(ret_code, "hbDNNReleaseTask failed");
    // == time-consuming test ==
    auto after_infer = std::chrono::system_clock::now();
    auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(after_infer - before_infer).count();
    RCLCPP_INFO_STREAM(rclcpp::get_logger("OccnetInfer"), "=> time cost: " << interval << " ms, fps: " << 1 / (interval / 1000.0));

    ret_code = postprocess();
    HB_CHECK_SUCCESS(ret_code, "postprocess failed");

    return ret_code;
}

int OccNetInfer::postprocess()
{
    int ret_code = 0;
    RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> ----- postprocess -----");
    occ_grids_.clear();
    hbDNNTensor output_tensor = output_tensors_[0];
    if (output_tensor.properties.tensorType != HB_DNN_TENSOR_TYPE_F32)
    {
        return -1;
    }

    auto output_tensor_addr = reinterpret_cast<float *>(output_tensor.sysMem[0].virAddr);
    int dims = output_tensor.properties.validShape.numDimensions;
    RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> output tensor dims: %d", dims);
    if (dims != 5)
    {
        return -1;
    }
    int *shape = output_tensor.properties.validShape.dimensionSize;
    int B = shape[0], N = shape[1], X = shape[2], Y = shape[3], Z = shape[4];
    RCLCPP_INFO(rclcpp::get_logger("OccnetInfer"), "=> output tensor shape: [%d, %d, %d, %d, %d]", B, N, X, Y, Z);
    for (int b = 0; b < B; b++)
    {
        for (int x = 0; x < X; x++)
        {
            for (int y = 0; y < Y; y++)
            {
                for (int z = 0; z < Z; z++)
                {
                    OccGrid occ_grid;
                    occ_grid.X = x;
                    occ_grid.Y = y;
                    occ_grid.Z = z;
                    for (int n = 0; n < N; n++)
                    {
                        int index = b * N * X * Y * Z + n * X * Y * Z + x * Y * Z + y * Z + z;
                        float occ_val = output_tensor_addr[index];
                        occ_grid.occ_probs.push_back(occ_val);
                    }
                    occ_grids_.push_back(occ_grid);
                }
            }
        }
    }
    return ret_code;
}

int OccNetInfer::save_result_to_txt(const std::string &txt_file_path)
{
    std::ofstream file(txt_file_path); // create and open the file
    if (!file.is_open())
    {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("OccnetInfer"), "=> Unable to open file: " << txt_file_path);
        return -1;
    }

    // iterate over each OccGrid
    for (const OccGrid &occ_grid : occ_grids_)
    {
        file << occ_grid.X << ", " << occ_grid.Y << ", " << occ_grid.Z << ", ";
        for (size_t i = 0; i < occ_grid.occ_probs.size(); ++i)
        {
            file << occ_grid.occ_probs[i];
            if (i < occ_grid.occ_probs.size() - 1)
            {
                file << ", ";
            }
        }
        file << std::endl;
    }

    file.close(); // close the file
    return 0;
}

std::string OccNetInfer::tensor_type_to_str(int32_t tensor_type)
{
    switch (tensor_type)
    {
    case HB_DNN_IMG_TYPE_Y:
        return "HB_DNN_IMG_TYPE_Y";
    case HB_DNN_IMG_TYPE_NV12:
        return "HB_DNN_IMG_TYPE_NV12";
    case HB_DNN_IMG_TYPE_NV12_SEPARATE:
        return "HB_DNN_IMG_TYPE_NV12_SEPARATE";
    case HB_DNN_IMG_TYPE_YUV444:
        return "HB_DNN_IMG_TYPE_YUV444";
    case HB_DNN_IMG_TYPE_RGB:
        return "HB_DNN_IMG_TYPE_RGB";
    case HB_DNN_IMG_TYPE_BGR:
        return "HB_DNN_IMG_TYPE_BGR";
    case HB_DNN_TENSOR_TYPE_S4:
        return "HB_DNN_TENSOR_TYPE_S4";
    case HB_DNN_TENSOR_TYPE_U4:
        return "HB_DNN_TENSOR_TYPE_U4";
    case HB_DNN_TENSOR_TYPE_S8:
        return "HB_DNN_TENSOR_TYPE_S8";
    case HB_DNN_TENSOR_TYPE_U8:
        return "HB_DNN_TENSOR_TYPE_U8";
    case HB_DNN_TENSOR_TYPE_F16:
        return "HB_DNN_TENSOR_TYPE_F16";
    case HB_DNN_TENSOR_TYPE_S16:
        return "HB_DNN_TENSOR_TYPE_S16";
    case HB_DNN_TENSOR_TYPE_U16:
        return "HB_DNN_TENSOR_TYPE_U16";
    case HB_DNN_TENSOR_TYPE_F32:
        return "HB_DNN_TENSOR_TYPE_F32";
    case HB_DNN_TENSOR_TYPE_S32:
        return "HB_DNN_TENSOR_TYPE_S32";
    case HB_DNN_TENSOR_TYPE_U32:
        return "HB_DNN_TENSOR_TYPE_U32";
    case HB_DNN_TENSOR_TYPE_F64:
        return "HB_DNN_TENSOR_TYPE_F64";
    case HB_DNN_TENSOR_TYPE_S64:
        return "HB_DNN_TENSOR_TYPE_S64";
    case HB_DNN_TENSOR_TYPE_U64:
        return "HB_DNN_TENSOR_TYPE_U64";
    case HB_DNN_TENSOR_TYPE_MAX:
        return "HB_DNN_TENSOR_TYPE_MAX";
    default:
        return "Unknown";
    }
}