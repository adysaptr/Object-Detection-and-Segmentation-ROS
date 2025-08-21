// ########################################################################################################
// ####################                          PROGRAM DEMO                          #################### 
// ########################################################################################################

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    ros::init(argc, argv, "camera_publisher_node");
    ros::NodeHandle nh;
    ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("/camera", 10);
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        ROS_ERROR("Kamera tidak dapat diakses!");
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;
    ros::Rate loop_rate(30); // 30 FPS
    while (ros::ok()) {
        cap >> frame;
        if (frame.empty()) {
            ROS_WARN("Empty frame captured!");
            continue;
        }
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = ros::Time::now();
        image_pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}