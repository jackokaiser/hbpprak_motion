
@nrp.MapSpikeSink("output1", nrp.brain.actors[0], nrp.population_rate)
@nrp.MapSpikeSink("output2", nrp.brain.actors[1], nrp.population_rate)
@nrp.MapSpikeSink("output3", nrp.brain.actors[2], nrp.population_rate)
@nrp.MapSpikeSink("output4", nrp.brain.actors[3], nrp.population_rate)
@nrp.MapSpikeSink("output5", nrp.brain.actors[4], nrp.population_rate)
@nrp.MapSpikeSink("output6", nrp.brain.actors[5], nrp.population_rate)
@nrp.MapSpikeSink("debug", nrp.brain.sensors, nrp.population_rate)
@nrp.MapRobotPublisher('arm_1_joint', Topic('/robot/hollie_real_left_arm_1_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('arm_2_joint', Topic('/robot/hollie_real_left_arm_2_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('arm_3_joint', Topic('/robot/hollie_real_left_arm_3_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('arm_4_joint', Topic('/robot/hollie_real_left_arm_4_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('arm_5_joint', Topic('/robot/hollie_real_left_arm_5_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('arm_6_joint', Topic('/robot/hollie_real_left_arm_6_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.Neuron2Robot()
# Example TF: get output neuron voltage and actuate the arm with the current simulation time. You could do something with the voltage here and command the robot accordingly.
def swing(t, debug, output1, output2, output3, output4, output5, output6, arm_1_joint, arm_2_joint, arm_3_joint, arm_4_joint, arm_5_joint, arm_6_joint):
    outputs = [min(out.rate / 100.0, 1.0) for out in (output1, output2, output3, output4, output5, output6)]
    arm_1_joint.send_message(std_msgs.msg.Float64(outputs[0] * np.pi))  # rotate 0 - pi
    arm_2_joint.send_message(std_msgs.msg.Float64(outputs[1] * np.pi / 2.0))  # bend   0 - pi/2
    arm_3_joint.send_message(std_msgs.msg.Float64(outputs[2] * 2 - 1))  # bend  -1 - 1
    arm_4_joint.send_message(std_msgs.msg.Float64(outputs[3] * np.pi - np.pi / 2))  # rotate -pi / 2 - pi / 2
    arm_5_joint.send_message(std_msgs.msg.Float64(outputs[4] * 2 - 1))  # bend  -1 - 1
    arm_6_joint.send_message(std_msgs.msg.Float64(outputs[5] * np.pi))  # rotate 0 - pi

