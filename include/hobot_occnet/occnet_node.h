#ifndef OCCNET_NODE_H
#define OCCNET_NODE_H

#include "rclcpp/rclcpp.hpp"
#include "hobot_occnet/img_convert_utils.h"
#include <filesystem>
#include <fstream>
#include "sensor_msgs/msg/image.hpp"
#include "dnn/hb_dnn.h"
#include "opencv2/opencv.hpp"

namespace fs = std::filesystem;

class OccNetNode : public rclcpp::Node
{
public:
    explicit OccNetNode(const std::string &node_name = "hobot_occnet_node", const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions());
    ~OccNetNode() = default;

private:
    // ============================================================ callback fun ============================================================
    /**
     * @brief occupancy network infer callback fun
     * @param stereo_msg stereo images ros message
     */
    void infer_callback(const sensor_msgs::msg::Image::ConstSharedPtr &stereo_msg);

    void infer_test();
    int prepare_input_tensor(std::vector<hbDNNTensor> &input_tensors, hbDNNHandle_t dnn_handle);
    int prepare_output_tensor(std::vector<hbDNNTensor> &output_tensors, hbDNNHandle_t dnn_handle);
    int postprocess(std::vector<hbDNNTensor> &output_tensors, std::string filename);

    // ============================================================ member variables ========================================================
    std::string occ_model_file_path_;
    std::string local_image_dir_;
};

#endif