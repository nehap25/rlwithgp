full_pose, robot, objects,use_gui,goal_pose,mass_bool=setup(True)
for action in actions:
  robot,objects,use_gui,reward=step(action[0],action[1],robot,objects,goal_pose,use_gui)
  print(reward)
