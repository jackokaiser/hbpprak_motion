
@nrp.MapSpikeSource("input0", nrp.brain.sensors[0], nrp.dc_source)
@nrp.MapSpikeSource("input1", nrp.brain.sensors[1], nrp.dc_source)
@nrp.MapSpikeSource("input2", nrp.brain.sensors[2], nrp.dc_source)
@nrp.MapRobotSubscriber('model_states', Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.Robot2Neuron()
def set_neuron_voltage(t, input0, input1, input2, model_states):
    states = model_states.value
    input0.amplitude = 0.0
    input1.amplitude = 0.0
    input2.amplitude = 0.0
    try:
        i = states.name.index("ball")
        y = states.pose[i].position.y
        if y > 4.0 or y < 0.0:
            input2.amplitude = 2.0
        elif y > 2.0:
            input1.amplitude = 2.0
        else:
            input0.amplitude = 2.0
    except (ValueError, AttributeError):
        input2.amplitude = 2.0
