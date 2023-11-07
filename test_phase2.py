import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from env import sanwo
from utility_p2 import helper, rewards
from collections import namedtuple, deque
from itertools import count
import torch

env = sanwo()
helper_obj=helper()
rewards_obj=rewards()



angles_dict = {
    0: -np.pi,
    1: -3*np.pi/4,
    2: -np.pi/2,
    3: -np.pi/4,
    4: 0,
    5: np.pi/4,
    6: np.pi/2,
    7: 3*np.pi/4
}

# Configurations
xml_path = 'sanwo_p2.xml' #xml file (assumes this is in the same folder as this file)
simend = 20#simulation time
print_camera_config = 0 #set to 1 to print camera config
						#this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0




# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                		# MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options
def render_it(i):
	mj.mj_resetData(model, data)
	mj.mj_forward(model, data)
	start_agent_x, start_agent_y, ball_x, ball_y = helper_obj.generate_coordinates(i, random_start=True)
	agentcoord=(start_agent_x, start_agent_y)
	ballcoord=(ball_x, ball_y)
	data.qpos[:2]=[start_agent_x, start_agent_y]
	data.qpos[28:30]=[ball_x, ball_y]
	targetcoord=helper_obj.agents_point(data.qpos[14:16], data.qpos[21:23], [ball_x, ball_y], 0.5)
	#print(targetcoord)
	target_x, target_y=targetcoord
	
	data.qpos[7:9]=[target_x, target_y]
	#print("target", target_x, target_y)
	relative_target_x, relative_target_y=helper_obj.find_relative_target_coordinates(agentcoord, targetcoord)
	state=np.array([start_agent_x, start_agent_y, relative_target_x, relative_target_y])
	state=torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
	#print(state)
	done=False
	# Init GLFW, create window, make OpenGL context current, request v-sync
	glfw.init()
	window = glfw.create_window(1200, 900, 'Maze problem', None, None)
	glfw.make_context_current(window)
	glfw.swap_interval(1)

	# initialize visualization data structures
	mj.mjv_defaultCamera(cam)
	mj.mjv_defaultOption(opt)
	scene = mj.MjvScene(model, maxgeom=10000)
	context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    # Callback functions
	def keyboard(window, key, scancode, act, mods):
		if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
			mj.mj_resetData(model, data)
			mj.mj_forward(model, data)

	def mouse_button(window, button, act, mods):
		# update button state
		global button_left
		global button_middle
		global button_right

		button_left = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
		button_middle = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
		button_right = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

		# update mouse position
		glfw.get_cursor_pos(window)

	def mouse_move(window, xpos, ypos):
		# compute mouse displacement, save
		global lastx
		global lasty
		global button_left
		global button_middle
		global button_right

		dx = xpos - lastx
		dy = ypos - lasty
		lastx = xpos
		lasty = ypos


		# no buttons down: nothing to do
		if (not button_left) and (not button_middle) and (not button_right):
			return

		# get current window size
		width, height = glfw.get_window_size(window)

		# get shift key state
		PRESS_LEFT_SHIFT = glfw.get_key(
			window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
		PRESS_RIGHT_SHIFT = glfw.get_key(
			window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
		mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

		# determine action based on mouse button
		if button_right:
			if mod_shift:
				action = mj.mjtMouse.mjMOUSE_MOVE_H
			else:
				action = mj.mjtMouse.mjMOUSE_MOVE_V
		elif button_left:
			if mod_shift:
				action = mj.mjtMouse.mjMOUSE_ROTATE_H
			else:
				action = mj.mjtMouse.mjMOUSE_ROTATE_V
		else:
			action = mj.mjtMouse.mjMOUSE_ZOOM

		mj.mjv_moveCamera(model, action, dx/height,
						dy/height, scene, cam)

	def scroll(window, xoffset, yoffset):
		action = mj.mjtMouse.mjMOUSE_ZOOM
		mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

	# install GLFW mouse and keyboard callbacks
	glfw.set_key_callback(window, keyboard)
	glfw.set_cursor_pos_callback(window, mouse_move)
	glfw.set_mouse_button_callback(window, mouse_button)
	glfw.set_scroll_callback(window, scroll)

	cam.azimuth = 90.17327778974852
	cam.elevation = -88.52113309352517
	cam.distance =  13.195704294990568
	cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])
	score=[]
	while not glfw.window_should_close(window):
		time_prev = data.time

		while (data.time - time_prev < 1/60.0) and state is not None:
			mj.mj_step(model, data)
			#print(state)
			#angle, speed = env.action_space.sample()
			
			# Select an action using the agent's policy
			action = helper_obj.find_action(state) #policy_net(state).argmax(dim=1)
			#print(action)
			#observation, reward, terminated, truncated, _ = env.step(action.item())
			#action = policy_net(state).argmin(dim=1).item()
			#print(action)
			#angle=action_mapping[action]
			direction=helper_obj.move_and_rotate(data.xpos[:2], action)
			direction = np.array(direction)
			direction /= np.linalg.norm(direction)  # normalize the velocity vector
			data.qvel[:2] = 0.5 * direction
			#print(data.qpos[:2])
			agentcoord, targetcoord=data.qpos[:2], data.qpos[7:9]
			done, Termination, reward, relative_target_x, relative_target_y=rewards_obj.cal_reward(relative_target_x, relative_target_y, agentcoord, targetcoord)
			reward=torch.tensor([reward], device="cpu")
			if done==True:next_state=None
			else:next_state=torch.tensor(np.array([agentcoord[0], agentcoord[1], relative_target_x, relative_target_y]), dtype=torch.float32, device="cpu").unsqueeze(0)
			state = next_state

		# End simulation based on time
		if (data.time>=simend) or Termination:
			break

		# get framebuffer viewport
		viewport_width, viewport_height = glfw.get_framebuffer_size(window)
		viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

		#print camera configuration (help to initialize the view)
		if (print_camera_config==1):
			print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
			print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

		# Update scene and render
		mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
		mj.mjr_render(viewport, scene, context)

		# swap OpenGL buffers (blocking call due to v-sync)
		glfw.swap_buffers(window)

		# process pending GUI events, call GLFW callbacks
		glfw.poll_events()
		#torch.save(policy_net.state_dict(), 'dqn_model.pth')
	glfw.terminate()
	return sum(score), score

#env.action_space.sample()
for i in range(10):
	print(render_it(i))
