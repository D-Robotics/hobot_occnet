#ifndef OCCNET_INFER_H
#define OCCNET_INFER_H

#include <fstream> 
#include "rclcpp/rclcpp.hpp"
#include "dnn/hb_dnn.h"
#include "opencv2/opencv.hpp"
#include "hobot_occnet/img_convert_utils.h"

// =================================================================================================================================
#define HB_CHECK_SUCCESS(ret_code, errmsg)                                                                                                                                                             \
    do                                                                                                                                                                                                 \
    {                                                                                                                                                                                                  \
        /*value can be call of function*/                                                                                                                                                              \
        if (ret_code != 0)                                                                                                                                                                             \
        {                                                                                                                                                                                              \
            RCLCPP_ERROR_STREAM(rclcpp::get_logger("OccnetInfer"), "=> [BPU ERROR]: " << errmsg << ", error code: " << ret_code);                                                                      \
        }                                                                                                                                                                                              \
    } while (0);

// =================================================================================================================================
/**
 * @brief input image type
 */
enum class InputImgType
{
    BGR8,
    NV12
};

// =================================================================================================================================
/**
 * @brief Save Occnet infer result as OccGrid
 */
struct OccGrid
{
    int X;                        // grid X coord
    int Y;                        // grid Y coord
    int Z;                        // grid Z coord
    std::vector<float> occ_probs; // Ooccupancy probability for each label
};

// =================================================================================================================================
/**
 * @brief Load Occnet and infer
 */
class OccNetInfer
{

public:
    OccNetInfer() = default;
    ~OccNetInfer() = default;

    int init(std::string &occ_model_file_path);

    /**
     * @brief infer by occ model
     */
    int forward(const cv::Mat &left_img, const cv::Mat &right_img, const InputImgType &input_img_type);

    /**
     * save result to txt file
     */
    int save_result_to_txt(const std::string &txt_file_path);

private:
    // ===================================== func =======================================
    /**
     * @brief allocate memory for input tensors, nv12 format
     */
    int prepare_input_tensor_nv12();

    /**
     * @brief allocate memory for output tensors
     */
    int prepare_output_tensor();

    /**
     * @brief fill data to input tensor
     */
    int fill_bgr_to_tensor_nv12(const cv::Mat &left_img_bgr, const cv::Mat &right_img_bgr);

    /**
     * @brief postprocess
     */
    int postprocess();

    std::string tensor_type_to_str(int32_t tensor_type);

    // ===================================== member =====================================
    /** init */
    std::string occ_model_file_path_;
    hbPackedDNNHandle_t packed_dnn_handle_;
    const char **model_name_list_;
    int model_count_ = 0;
    hbDNNHandle_t dnn_handle_;
    int input_count_ = 0;
    int output_count_ = 0;

    /** tensor */
    std::vector<hbDNNTensor> input_tensors_;
    std::vector<hbDNNTensor> output_tensors_;
    int model_input_h_;
    int model_input_w_;
    int32_t input_tensor_type_;

    /** result */
    std::vector<OccGrid> occ_grids_;
};

#endif