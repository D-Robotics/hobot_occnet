#ifndef OCCNET_NODE_H
#define OCCNET_NODE_H

#include "rclcpp/rclcpp.hpp"
#include <filesystem>
#include "hobot_occnet/occnet_infer.h"
#include "sensor_msgs/msg/image.hpp"
#include "opencv2/opencv.hpp"

namespace fs = std::filesystem;

/**
 * @brief Occnet ros process node
 */
class OccNetNode : public rclcpp::Node
{
public:
    explicit OccNetNode(const std::string &node_name = "hobot_occnet_node", const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions());
    ~OccNetNode() = default;

private:
    // ===================================== callback fun ===============================
    /**
     * @brief occupancy network online infer callback fun
     * @param stereo_msg stereo images ros message
     */
    void infer_callback(const sensor_msgs::msg::Image::ConstSharedPtr &stereo_msg);

    /**
     * @brief occupancy network offline infer fun
     */
    void infer_offline();

    // ===================================== member =====================================
    std::string occ_model_file_path_;

    bool use_local_image_;
    std::string local_image_dir_;

    OccNetInfer occnet_infer_;
};

#endif