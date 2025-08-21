
(cl:in-package :asdf)

(defsystem "time_sync_pkg-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "SyncData" :depends-on ("_package_SyncData"))
    (:file "_package_SyncData" :depends-on ("_package"))
  ))