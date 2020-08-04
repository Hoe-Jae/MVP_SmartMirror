import argparse

args = argparse.ArgumentParser(description='MVP Capstone Project')

args.add_argument('--depth_width',  dest='dWidth', type=int, help='Depth Camera Width [640]', default=640)
args.add_argument('--depth_height', dest='dHeight', type=int, help='Depth Camera Height [480]', default=480)
args.add_argument('--bkgnd_cnt', dest='dBackCnt', type=int, help='Depth Camera Background Cnt [100]', default=100)
args.add_argument('--bkgnd_default', dest='dBackDef', type=int, help='Depth Camera Background Default [1500]', default=1500)

args.add_argument('--rgb_width',   dest='rWidth', type=int, help='RGB Camera Width [1280]', default=1280)
args.add_argument('--rgb_height',  dest='rHeight', type=int, help='RGB Camera Height [720]', default=720)

args.add_argument('--MOTOR_TOP',  dest='mTop', type=int, help='Step Motor Top Pin index [29]', default=29)
args.add_argument('--MOTOR_MID',  dest='mMid', type=int, help='Step Motor Mid Pin index [31]', default=31)
args.add_argument('--MOTOR_BOT',  dest='mBot', type=int, help='Step Motor Bot Pin index [33]', default=33)

args.add_argument('--TCP_Socket',  dest='tcpPort', type=int, help='TCP Port [5319]', default=5319)
args.add_argument('--num_classes',  dest='numClasses', type=int, help='YOLOv3 number of Classes [15]', default=15)

args.add_argument('--show', dest='show', help='Camera Image Visualization', action='store_true')

args = args.parse_args()
print(args)
