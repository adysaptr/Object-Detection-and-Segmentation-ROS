;; Auto-generated. Do not edit!


(when (boundp 'time_sync_pkg::SyncData)
  (if (not (find-package "TIME_SYNC_PKG"))
    (make-package "TIME_SYNC_PKG"))
  (shadow 'SyncData (find-package "TIME_SYNC_PKG")))
(unless (find-package "TIME_SYNC_PKG::SYNCDATA")
  (make-package "TIME_SYNC_PKG::SYNCDATA"))

(in-package "ROS")
;;//! \htmlinclude SyncData.msg.html
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))


(defclass time_sync_pkg::SyncData
  :super ros::object
  :slots (_image _scan ))

(defmethod time_sync_pkg::SyncData
  (:init
   (&key
    ((:image __image) (instance sensor_msgs::Image :init))
    ((:scan __scan) (instance sensor_msgs::LaserScan :init))
    )
   (send-super :init)
   (setq _image __image)
   (setq _scan __scan)
   self)
  (:image
   (&rest __image)
   (if (keywordp (car __image))
       (send* _image __image)
     (progn
       (if __image (setq _image (car __image)))
       _image)))
  (:scan
   (&rest __scan)
   (if (keywordp (car __scan))
       (send* _scan __scan)
     (progn
       (if __scan (setq _scan (car __scan)))
       _scan)))
  (:serialization-length
   ()
   (+
    ;; sensor_msgs/Image _image
    (send _image :serialization-length)
    ;; sensor_msgs/LaserScan _scan
    (send _scan :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; sensor_msgs/Image _image
       (send _image :serialize s)
     ;; sensor_msgs/LaserScan _scan
       (send _scan :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; sensor_msgs/Image _image
     (send _image :deserialize buf ptr-) (incf ptr- (send _image :serialization-length))
   ;; sensor_msgs/LaserScan _scan
     (send _scan :deserialize buf ptr-) (incf ptr- (send _scan :serialization-length))
   ;;
   self)
  )

(setf (get time_sync_pkg::SyncData :md5sum-) "149818d6ea321fa3e69c3870cecbc7b3")
(setf (get time_sync_pkg::SyncData :datatype-) "time_sync_pkg/SyncData")
(setf (get time_sync_pkg::SyncData :definition-)
      "sensor_msgs/Image image
sensor_msgs/LaserScan scan
================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: sensor_msgs/LaserScan
# Single scan from a planar laser range-finder
#
# If you have another ranging device with different behavior (e.g. a sonar
# array), please find or create a different message, since applications
# will make fairly laser-specific assumptions about this data

Header header            # timestamp in the header is the acquisition time of 
                         # the first ray in the scan.
                         #
                         # in frame frame_id, angles are measured around 
                         # the positive Z axis (counterclockwise, if Z is up)
                         # with zero angle being forward along the x axis
                         
float32 angle_min        # start angle of the scan [rad]
float32 angle_max        # end angle of the scan [rad]
float32 angle_increment  # angular distance between measurements [rad]

float32 time_increment   # time between measurements [seconds] - if your scanner
                         # is moving, this will be used in interpolating position
                         # of 3d points
float32 scan_time        # time between scans [seconds]

float32 range_min        # minimum range value [m]
float32 range_max        # maximum range value [m]

float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)
float32[] intensities    # intensity data [device-specific units].  If your
                         # device does not provide intensities, please leave
                         # the array empty.

")



(provide :time_sync_pkg/SyncData "149818d6ea321fa3e69c3870cecbc7b3")


