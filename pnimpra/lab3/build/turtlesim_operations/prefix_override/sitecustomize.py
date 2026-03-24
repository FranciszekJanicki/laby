import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/student/Documents/pnimpra/lab3/ros2_ws/install/turtlesim_operations'
