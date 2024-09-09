#include "rclcpp/rclcpp.hpp"
#include "hobot_occnet/occnet_node.h"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OccNetNode>("hobot_occnet_node", rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}