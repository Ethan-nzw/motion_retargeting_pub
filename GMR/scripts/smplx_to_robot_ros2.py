import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import threading


class RobotJointPublisher(Node):
    def __init__(self):
        super().__init__('motion_retargeting_node')
        self.publisher = self.create_publisher(
            Float64MultiArray, 
            '/jointspace_commands_Dual', 
            10
        )
        self.get_logger().info('Robot Joint Publisher initialized')

    def publish_joint_angles(self, joint_angles):
        msg = Float64MultiArray()
        msg.data = joint_angles
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published joint angles: {joint_angles}')

def main():

    rclpy.init()
    
    # Create publisher node
    joint_publisher = RobotJointPublisher()
    
    # Run ROS2 node in a separate thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(joint_publisher,), daemon=True)
    ros_thread.start()

    try:
        smplx_file="/home/eai/motion_retarget/GMR/motion_data/speech_motion/test_speech_motion_smplx6.npz"
        robot_type = "tienkungpro"
        # choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
        #              "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
        #             "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
        #             "pnd_adam_lite", "openloong", "tienkung"],

        HERE = pathlib.Path(__file__).parent
        SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
        
        # Load SMPLX trajectory
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            smplx_file, SMPLX_FOLDER
        )

        # align fps
        tgt_fps = 30
        smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
        
    
        # Initialize the retargeting system
        retarget = GMR(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=robot_type,
        )
        
        i = 0
        while rclpy.ok():
            # while True:
            i += 1
            if i >= len(smplx_data_frames):
                break
                
            #     # FPS measurement
            #     fps_counter += 1
            #     current_time = time.time()
            #     if current_time - fps_start_time >= fps_display_interval:
            #         actual_fps = fps_counter / (current_time - fps_start_time)
            #         print(f"Actual rendering FPS: {actual_fps:.2f}")
            #         fps_counter = 0
            #         fps_start_time = current_time
                
            # Update task targets.
            smplx_data = smplx_data_frames[i]

            # retarget
            qpos = retarget.retarget(smplx_data)
            # print("Retargeted qpos:", qpos)
            # joints_publish_left = float(qpos[-8:])
            joints_publish_left = [float(x) for x in qpos[-14:]]
            # print("joints to publish left:", joints_publish_left)

            joint_names = retarget.robot_motor_names
            # print("joint names: ", joint_names.keys())

            # Publish to ROS2 topic
            joint_publisher.publish_joint_angles(joints_publish_left)
            time.sleep(0.05)
    
    except Exception as e:
        print(f"Error during retargeting: {e}")
    finally:
        # Cleanup
        joint_publisher.destroy_node()
        rclpy.shutdown()
            
if __name__ == "__main__":
    main()
      
