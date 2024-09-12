#include "hobot_occnet/occnet_node.h"
#include <cassert>

OccNetNode::OccNetNode(const std::string &node_name, const rclcpp::NodeOptions &node_options) : Node(node_name, node_options)
{
    RCLCPP_INFO(this->get_logger(), "=> init %s", node_name.c_str());
    // =================================================================================================================================
    this->declare_parameter("occ_model_file_path", "");
    this->declare_parameter("use_local_image", false);
    this->declare_parameter("local_image_dir", "");
    this->get_parameter("occ_model_file_path", occ_model_file_path_);
    this->get_parameter("use_local_image", use_local_image_);
    this->get_parameter("local_image_dir", local_image_dir_);
    RCLCPP_INFO_STREAM(this->get_logger(), "\033[31m=> occ_model_file_path: " << occ_model_file_path_ << "\033[0m");

    // =================================================================================================================================
    occnet_infer_.init(occ_model_file_path_);
    if (use_local_image_)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "\033[31m=> local_image_dir: " << local_image_dir_ << "\033[0m");
        infer_offline();
    }
    else
    {
    }
}

void OccNetNode::infer_callback(const sensor_msgs::msg::Image::ConstSharedPtr &stereo_msg)
{
}

void OccNetNode::infer_offline()
{
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
            occnet_infer_.forward(left_img_bgr, right_img_bgr, InputImgType::BGR8);
        }
    }
}