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

// const std::vector<std::string> CLASS_NAMES = {
//     "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
//     "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
//     "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
//     "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
//     "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
//     "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
//     "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
//     "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
//     "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
//     "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
//     "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
//     "teddy bear",     "hair drier", "toothbrush"};

// const std::vector<std::vector<unsigned int>> COLORS = {
//     {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
//     {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
//     {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
//     {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
//     {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
//     {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
//     {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
//     {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
//     {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
//     {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
//     {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
//     {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
//     {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
//     {80, 183, 189},  {128, 128, 0}};

// const std::vector<std::vector<unsigned int>> MASK_COLORS = {
//     {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
//     {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
//     {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};

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

        // for (size_t i = 0; i < scan.ranges.size(); ++i) {
        //     float range = scan.ranges[i];
        //     if (range < range_min || range > range_max) continue;

        //     // Konversi ke koordinat LiDAR
        //     double angle = angle_min + i * angle_increment;
        //     double x = range * cos(angle);
        //     double y = range * sin(angle);
        //     double z = 0.0;

        //     // Transformasi ke frame kamera
        //     Eigen::Vector3d P_L(x, y, z);
        //     Eigen::Vector3d P_C = R * P_L + T;

        //     // Proyeksi ke gambar
        //     Eigen::Vector3d P_I = K * P_C;
        //     if (P_I(2) <= 0) continue;
        //     int u = static_cast<int>(P_I(0) / P_I(2));
        //     int v = static_cast<int>(P_I(1) / P_I(2));

        //     if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
        //         lidar_points.push_back(LidarPoint{u, v, range, angle});
        //         //cv::circle(image, cv::Point(u, v), 2, cv::Scalar(0, 0, 255), -1);
        //     }
        // }

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





// ==============================================================================================================================


// #include "opencv2/opencv.hpp"
// #include "yolov8-seg.hpp"
// #include <ros/ros.h>
// #include <cv_bridge/cv_bridge.h>
// #include <sensor_msgs/Image.h>
// #include <sensor_msgs/LaserScan.h>
// #include <Eigen/Dense>
// #include <chrono>
// #include <mutex>
// #include <iomanip>

// const std::vector<std::string> CLASS_NAMES = {
//     "Bus", "Car", "Motorcycle", "Person", "Truck"};

// const std::vector<std::vector<unsigned int>> COLORS = {
//     {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48}};

// const std::vector<std::vector<unsigned int>> MASK_COLORS = {
//     {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49}};

// // const std::vector<std::string> CLASS_NAMES = {
// //     "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
// //     "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
// //     "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
// //     "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
// //     "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
// //     "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
// //     "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
// //     "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
// //     "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
// //     "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
// //     "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
// //     "teddy bear",     "hair drier", "toothbrush"};

// // const std::vector<std::vector<unsigned int>> COLORS = {
// //     {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
// //     {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
// //     {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
// //     {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
// //     {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
// //     {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
// //     {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
// //     {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
// //     {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
// //     {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
// //     {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
// //     {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
// //     {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
// //     {80, 183, 189},  {128, 128, 0}};

// // const std::vector<std::vector<unsigned int>> MASK_COLORS = {
// //     {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
// //     {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
// //     {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};

// class YOLOv8SegNode{
//     public:
//     YOLOv8SegNode(ros::NodeHandle& nh, const std::string& engine_path) : nh_(nh), engine_path_(engine_path) {
//         image_sub_ = nh_.subscribe("/camera_sync", 1, &YOLOv8SegNode::imageCallback, this);
//         scan_sub_ = nh_.subscribe("/scan_sync", 1, &YOLOv8SegNode::scanCallback, this);
//         yolov8_seg_ = new YOLOv8_seg(engine_path_);
//         yolov8_seg_->make_pipe(true);

//         fx = 939.988307; fy = 945.520654; cx = 337.746494; cy = 221.072054;
//         K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
//         R << -0.04006809,  0.99906152,  0.01645081,
//               0.02362902,  0.01740683, -0.99956924,
//              -0.99891752, -0.03966211, -0.0243043;
//         T << -0.01693666, 0.03729649, -0.06359766;
//     }

//     ~YOLOv8SegNode(){
//         delete yolov8_seg_;
//     }

//     private:
//     ros::NodeHandle nh_;
//     ros::Subscriber image_sub_;
//     ros::Subscriber scan_sub_;
//     std::string engine_path_;
//     YOLOv8_seg* yolov8_seg_;
//     sensor_msgs::LaserScanConstPtr latest_scan_;
//     std::mutex lidar_mutex_;

//     double fx, fy, cx, cy;
//     Eigen::Matrix3d K;
//     Eigen::Matrix3d R;
//     Eigen::Vector3d T;

//     struct LidarPoint {
//         int u;
//         int v;
//         float range;
//         float angle;
//     };

//     void scanCallback(const sensor_msgs::LaserScanConstPtr& msg) {
//         std::lock_guard<std::mutex> lock(lidar_mutex_);
//         latest_scan_ = msg;
//     }

//     void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
//         try {
//             cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
//             cv::Mat image = cv_ptr->image.clone();

//             cv::Mat res;
//             std::vector<Object> objs;
//             processInference(image, res, objs);

//             projectLidarToImage(res, objs);

//             cv::imshow("Detection & LiDAR Projection", res);
//             cv::waitKey(1);
//         } catch (cv_bridge::Exception& e) {
//             ROS_ERROR("cv_bridge exception: %s", e.what());
//         }
//     }

//     void processInference(cv::Mat& image, cv::Mat& res, std::vector<Object>& objs) {
//         cv::Size size = cv::Size{640, 640};
//         int topk = 100;
//         int seg_h = 160;
//         int seg_w = 160;
//         int seg_channels = 32;
//         float score_thres = 0.05f;
//         float iou_thres = 0.65f;

//         // Copy image to YOLOv8-seg
//         yolov8_seg_->copy_from_Mat(image, size);

//         // Inferensi
//         auto start = std::chrono::system_clock::now();
//         yolov8_seg_->infer();
//         auto end = std::chrono::system_clock::now();

//         // Postprocessing
//         yolov8_seg_->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);

//         // Gambar hasil deteksi
//         yolov8_seg_->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);

//         // Hitung FPS
//         double tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
//         double fps = 1000.0 / tc;

//         // Tampilkan FPS di layar
//         std::ostringstream oss;
//         oss << std::fixed << std::setprecision(2) << fps;
//         std::string fps_text = "FPS: " + oss.str();
//         cv::putText(res, fps_text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

//         // cv::imshow("Detection Window", res);
//         // cv::waitKey(1);
//     }

//     void projectLidarToImage(cv::Mat& image, const std::vector<Object>& objs) {
//         std::lock_guard<std::mutex> lock(lidar_mutex_);
//         if (!latest_scan_) return;

//         const auto& scan = *latest_scan_;
//         double angle_min = scan.angle_min;
//         double angle_increment = scan.angle_increment;
//         double range_min = scan.range_min;
//         double range_max = scan.range_max;

//         // Simpan titik Lidar
//         std::vector<LidarPoint> lidar_points;

//         for (size_t i = 0; i < scan.ranges.size(); ++i) {
//             float range = scan.ranges[i];
//             if (range < range_min || range > range_max) continue;

//             // Konversi ke koordinat LiDAR
//             double angle = angle_min + i * angle_increment;
//             double x = range * cos(angle);
//             double y = range * sin(angle);
//             double z = 0.0;

//             // Transformasi ke frame kamera
//             Eigen::Vector3d P_L(x, y, z);
//             Eigen::Vector3d P_C = R * P_L + T;

//             // Proyeksi ke gambar
//             Eigen::Vector3d P_I = K * P_C;
//             if (P_I(2) <= 0) continue;
//             int u = static_cast<int>(P_I(0) / P_I(2));
//             int v = static_cast<int>(P_I(1) / P_I(2));

//             if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
//                 lidar_points.push_back(LidarPoint{u, v, range, angle});
//                 //cv::circle(image, cv::Point(u, v), 2, cv::Scalar(0, 0, 255), -1);
//             }
//         }

//         // HITUNG JARAK BERDASARKAN TITIK DALAM BOUNDING BOX

//         // for (const auto& obj : objs) {
//         //     int x_min = obj.rect.x;
//         //     int y_min = obj.rect.y;
//         //     int x_max = obj.rect.x + obj.rect.width;
//         //     int y_max = obj.rect.y + obj.rect.height;

//         //     float min_distance = std::numeric_limits<float>::max(); // Inisialisasi dengan nilai maksimum
//         //     for (const auto& point : lidar_points) {
//         //         if (point.u >= x_min && point.u <= x_max && point.v >= y_min && point.v <= y_max) {
//         //             if (point.range < min_distance) {
//         //                 min_distance = point.range; // Update jarak terdekat
//         //             }

//         //             // Gambar titik Lidar hanya jika berada dalam bounding box objek
//         //             //cv::circle(image, cv::Point(point.u, point.v), 2, cv::Scalar(0, 0, 255), -1);
//         //         }
//         //     }

//         //     //Gambar jarak terdekat di dekat bounding box
//         //     if (min_distance != std::numeric_limits<float>::max()) { // Pastikan ada titik yang ditemukan
//         //         std::ostringstream oss;
//         //         oss << std::fixed << std::setprecision(2) << min_distance; // Presisi 2 digit desimal
//         //         std::string distance_text = "Dist: " + oss.str() + " m";
//         //         cv::putText(image, distance_text, cv::Point(x_min, y_min - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
//         //     }
//         // }


//         // HITUNG JARAK OBJEK BERDASARKAN TITIK DALAM MASK

//         for (const auto& obj : objs) {

//             if (obj.boxMask.empty()) continue;

//             std::vector<cv::Point> lidar_in_mask;
//             float min_distance_in_mask = std::numeric_limits<float>::max();
//             float min_angle_in_mask = 0.0;
//             bool found = false;

//             for (const auto& point : lidar_points) {
//                 int local_u = point.u - obj.rect.x;
//                 int local_v = point.v - obj.rect.y;

//                 if (local_u >= 0 && local_u < obj.boxMask.cols &&
//                     local_v >= 0 && local_v < obj.boxMask.rows) {

//                     if (obj.boxMask.at<uchar>(local_v, local_u) > 0) {
//                         // Simpan titik untuk diproses setelah loop
//                         lidar_in_mask.emplace_back(point.u, point.v);

//                         // Visualisasi titik
//                         // cv::circle(image, cv::Point(point.u, point.v), 2, cv::Scalar(0, 0, 255), -1);

//                         if (point.range < min_distance_in_mask) {
//                             min_distance_in_mask = point.range;
//                             min_angle_in_mask = point.angle; // Simpan sudut titik terdekat
//                             found = true;
//                         }

//                     }
//                 }
//             }

//             // Gambar jarak hanya jika ditemukan titik LiDAR dalam mask
//             if (found) {
//                 std::ostringstream oss;
//                 oss << std::fixed << std::setprecision(2);
//                 oss << "Dist: " << min_distance_in_mask << " m";
//                 std::string distance_text = oss.str();

//                 oss.str(""); // Reset stream
//                 oss << std::fixed << std::setprecision(2);
//                 oss << "Angle: " << min_angle_in_mask * 180.0 / M_PI << " deg";  // Konversi radian ke derajat
//                 std::string angle_text = oss.str();

//                 // Tampilkan kedua info
//                 // cv::putText(image, distance_text, cv::Point(obj.rect.x, obj.rect.y - 20), 
//                 //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//                 // cv::putText(image, angle_text, cv::Point(obj.rect.x, obj.rect.y - 5), 
//                 //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//             }

//             // === OUTPUT INFORMASI ===
//             int u_min = static_cast<int>(obj.rect.x);
//             int v_min = static_cast<int>(obj.rect.y);
//             int u_max = static_cast<int>(obj.rect.x + obj.rect.width);
//             int v_max = static_cast<int>(obj.rect.y + obj.rect.height);

//             std::cout << "\n--- Informasi Objek ---\n";
//             std::cout << "Bounding Box: u_min=" << u_min << ", v_min=" << v_min
//                     << ", u_max=" << u_max << ", v_max=" << v_max << "\n";
//             std::cout << "Jumlah Titik Lidar dalam Mask: " << lidar_in_mask.size() << "\n";

//             for (size_t i = 0; i < lidar_in_mask.size(); ++i) {
//                 std::cout << "Titik Lidar " << i+1 << ": u=" << lidar_in_mask[i].x
//                         << ", v=" << lidar_in_mask[i].y << "\n";
//             }


//             int x_min = obj.rect.x;
//             int y_min = obj.rect.y;
//             int x_max = obj.rect.x + obj.rect.width;
//             int y_max = obj.rect.y + obj.rect.height;

//             std::vector<std::pair<float, float>> distance_angle_pairs; // <range, angle>

//             for (size_t i = 0; i < scan.ranges.size(); ++i) {
//                 float range = scan.ranges[i];
//                 if (range < range_min || range > range_max) continue;

//                 double angle = angle_min + i * angle_increment;
//                 double x = range * cos(angle);
//                 double y = range * sin(angle);
//                 double z = 0.0;

//                 Eigen::Vector3d P_L(x, y, z);
//                 Eigen::Vector3d P_C = R * P_L + T;

//                 Eigen::Vector3d P_I = K * P_C;
//                 if (P_I(2) <= 0) continue;
//                 int u = static_cast<int>(P_I(0) / P_I(2));
//                 int v = static_cast<int>(P_I(1) / P_I(2));

//                 if (u >= x_min && u <= x_max && v >= y_min && v <= y_max) {
//                     distance_angle_pairs.emplace_back(range, angle); // simpan jarak & sudut
//                     cv::circle(image, cv::Point(u, v), 2, cv::Scalar(0, 0, 255), -1);
//                 }
//             }

//             // //Menampilkan informasi
//             // ROS_INFO("Objek %s di bbox [%d, %d, %d, %d] punya %lu titik lidar", 
//             //     CLASS_NAMES[obj.label].c_str(), x_min, y_min, x_max, y_max, distance_angle_pairs.size());

//             // for (const auto& [r, a] : distance_angle_pairs) {
//             //     float deg = a * 180.0f / M_PI;
//             //     ROS_INFO(" - Jarak: %.2f m, Sudut: %.2f derajat", r, deg);
//             // }
//         }

//     }
// };

// int main(int argc, char** argv) {
//     ros::init(argc, argv, "yolov8_seg_lidar_node");
//     ros::NodeHandle nh;
//     std::string engine_path = "/home/jetson/ACKERMAN/src/yolov8_seg_ros/model.engine";
//     YOLOv8SegNode node(nh, engine_path);
//     ros::spin();
//     return 0;
// }