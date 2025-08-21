// ########################################################################################################
// ####################                          PROGRAM DEMO                          #################### 
// ########################################################################################################


// ============================== OBJECT DETECTION AND DISTANCE ESTIMATION ================================



#include "opencv2/opencv.hpp"
#include "yolov8-seg.hpp"
#include <ros/ros.h>
#include <stdio.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <Eigen/Dense>
#include <chrono>
#include <mutex>
#include <iomanip>

const std::vector<std::string> CLASS_NAMES = {
    "Bus", "Car", "Motorcycle", "Person", "Truck"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49}};

class YOLOv8SegNode{
    public:
    YOLOv8SegNode(ros::NodeHandle& nh, const std::string& engine_path) : nh_(nh), engine_path_(engine_path) {
        image_sub_ = nh_.subscribe("/camera_sync", 1, &YOLOv8SegNode::imageCallback, this);
        scan_sub_ = nh_.subscribe("/scan_sync", 1, &YOLOv8SegNode::scanCallback, this);
        yolov8_seg_ = new YOLOv8_seg(engine_path_);
        yolov8_seg_->make_pipe(true);

        fx = 939.988307; fy = 945.520654; cx = 337.746494; cy = 221.072054;
        K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        R << -0.04006809,  0.99906152,  0.01645081,
              0.02362902,  0.01740683, -0.99956924,
             -0.99891752, -0.03966211, -0.0243043;
        T << -0.01693666, 0.03729649, -0.06359766;
    }

    ~YOLOv8SegNode(){
        delete yolov8_seg_;
    }

    private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Subscriber scan_sub_;
    std::string engine_path_;
    YOLOv8_seg* yolov8_seg_;
    sensor_msgs::LaserScanConstPtr latest_scan_;
    std::mutex lidar_mutex_;

    std::vector<std::string> log_lines_;

    double fx, fy, cx, cy;
    Eigen::Matrix3d K;
    Eigen::Matrix3d R;
    Eigen::Vector3d T;

    struct LidarPoint {
        int u;
        int v;
        float range;
        float angle;
    };

    void scanCallback(const sensor_msgs::LaserScanConstPtr& msg) {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        latest_scan_ = msg;
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat image = cv_ptr->image.clone();

            cv::Mat res;
            std::vector<Object> objs;
            processInference(image, res, objs);

            projectLidarToImage(res, objs);

            cv::imshow("Detection & LiDAR Projection", res);
            cv::waitKey(1);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void processInference(cv::Mat& image, cv::Mat& res, std::vector<Object>& objs) {
        cv::Size size = cv::Size{640, 640};
        int topk = 100;
        int seg_h = 160;
        int seg_w = 160;
        int seg_channels = 32;
        float score_thres = 0.25f;
        float iou_thres = 0.65f;

        // Copy image to YOLOv8-seg
        yolov8_seg_->copy_from_Mat(image, size);

        // Inferensi
        auto start = std::chrono::system_clock::now();
        yolov8_seg_->infer();
        auto end = std::chrono::system_clock::now();

        // Postprocessing
        yolov8_seg_->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);

        // Gambar hasil deteksi
        yolov8_seg_->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);

        // Hitung FPS
        double tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        double fps = 1000.0 / tc;

        // Tampilkan FPS di layar
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << fps;
        std::string fps_text = "FPS: " + oss.str();
        cv::putText(res, fps_text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Tampilkan semua log objek + jumlah titik di bawah FPS
        int y_offset = 45;  // Posisi vertikal awal
        for (const auto& line : log_lines_) {
            cv::putText(res, line, cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            y_offset += 20;  // Naikkan setiap baris teks
        }


        // Cetak ke terminal
        //ROS_INFO("Inference Time: %.2f ms | FPS: %.2f", tc, fps);
        //printf("%.2f\n", fps);


        // cv::imshow("Detection Window", res);
        // cv::waitKey(1);
    }

    void projectLidarToImage(cv::Mat& image, const std::vector<Object>& objs) {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        if (!latest_scan_) return;
        log_lines_.clear();

        const auto& scan = *latest_scan_;
        double angle_min = scan.angle_min;
        double angle_increment = scan.angle_increment;
        double range_min = scan.range_min;
        double range_max = scan.range_max;

        // Simpan titik Lidar
        std::vector<LidarPoint> lidar_points;

        for (size_t i = 0; i < scan.ranges.size(); ++i) {
            float range = scan.ranges[i];
            if (range < range_min || range > range_max) continue;

            double original_angle = angle_min + i * angle_increment;
            double adjusted_angle = M_PI - original_angle;  // Sesuaikan sudut

            // Normalisasi sudut ke rentang [-π, π]
            adjusted_angle = std::fmod(adjusted_angle, 2 * M_PI);
            if (adjusted_angle > M_PI) adjusted_angle -= 2 * M_PI;
            if (adjusted_angle < -M_PI) adjusted_angle += 2 * M_PI;

            // Hitung koordinat LiDAR menggunakan original_angle
            double x = range * cos(original_angle);
            double y = range * sin(original_angle);
            double z = 0.0;

            // Transformasi ke frame kamera
            Eigen::Vector3d P_L(x, y, z);
            Eigen::Vector3d P_C = R * P_L + T;

            // Proyeksi ke gambar
            Eigen::Vector3d P_I = K * P_C;
            if (P_I(2) <= 0) continue;
            int u = static_cast<int>(P_I(0) / P_I(2));
            int v = static_cast<int>(P_I(1) / P_I(2));

            if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
                lidar_points.push_back(LidarPoint{u, v, range, adjusted_angle});  // Simpan sudut yang disesuaikan
                // Visualisasi titik (opsional)
                // cv::circle(image, cv::Point(u, v), 2, cv::Scalar(0, 0, 255), -1);
            }
        }

        // HITUNG JARAK OBJEK BERDASARKAN TITIK DALAM MASK

        for (const auto& obj : objs) {
            if (obj.boxMask.empty()) continue;

            std::vector<std::pair<float, float>> points_in_mask; // (range, angle)
            int point_count = 0;

            for (const auto& point : lidar_points) {
                int local_u = point.u - obj.rect.x;
                int local_v = point.v - obj.rect.y;

                if (local_u >= 0 && local_u < obj.boxMask.cols &&
                    local_v >= 0 && local_v < obj.boxMask.rows) {

                    if (obj.boxMask.at<uchar>(local_v, local_u) > 0) {
                        // Simpan jarak dan sudut
                        points_in_mask.emplace_back(point.range, point.angle);
                        point_count++;

                        // Visualisasi titik
                        cv::circle(image, cv::Point(point.u, point.v), 2, cv::Scalar(0, 0, 255), -1);
                    }
                }
            }

            // Cetak jumlah dan detail titik di terminal
            if (!points_in_mask.empty()) {
                ROS_INFO(" ");
                ROS_INFO(" ");
                ROS_INFO("Objek: %s | Jumlah Titik : %d", CLASS_NAMES[obj.label].c_str(), point_count);

                // Tambahkan ke daftar log untuk ditampilkan di gambar
                std::ostringstream log_stream;
                log_stream << "Objek: " << CLASS_NAMES[obj.label] << " | Jumlah Titik: " << point_count;
                log_lines_.push_back(log_stream.str());


                for (size_t i = 0; i < points_in_mask.size(); ++i) {
                    float range_m = points_in_mask[i].first;
                    float angle_deg = points_in_mask[i].second * 180.0 / M_PI;

                    ROS_INFO(" Point #%zu | Dist: %.2f m | Angle: %.2f°", i + 1, range_m, angle_deg);
                }

                // Hitung rata-rata jarak dan sudut
                float total_distance = 0.0f;
                float total_angle = 0.0f;

                for (const auto& pt : points_in_mask) {
                    total_distance += pt.first;
                    total_angle += pt.second;
                }

                float avg_distance = total_distance / points_in_mask.size();
                float avg_angle = total_angle / points_in_mask.size();

                // Tampilkan rata-rata di layar
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2);
                oss << "Dist: " << avg_distance << " m";
                std::string distance_text = oss.str();

                oss.str("");
                oss << "Angle: " << (avg_angle * 180.0 / M_PI) << " deg";
                std::string angle_text = oss.str();

                //Gambar teks di gambar
                cv::putText(image, distance_text, cv::Point(obj.rect.x, obj.rect.y - 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                cv::putText(image, angle_text, cv::Point(obj.rect.x, obj.rect.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }

    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "yolov8_seg_lidar_node");
    ros::NodeHandle nh;
    std::string engine_path = "/home/jetson/ACKERMAN/src/yolov8_seg_ros/model.engine";
    // std::string engine_path = "/home/jetson/ACKERMAN/src/yolov8_seg_ros/yolov8n-seg.engine";
    YOLOv8SegNode node(nh, engine_path);
    ros::spin();
    return 0;
}
