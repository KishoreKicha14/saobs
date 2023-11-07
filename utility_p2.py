import math
import random
import numpy as np
class helper:
    def __init__(self):
        pass
    def move_and_rotate(self, current_coords, angle):
        x, y = current_coords
        x_prime = math.cos(angle)
        y_prime = math.sin(angle)
        return   [x_prime, y_prime]
    def find_slope(self, coord1, coord2):
        m=(coord1[1]-coord2[1])/(coord1[0]-coord2[0])
        c=coord1[1]-(m*coord1[0])
        return m, c

    def find_intersection(self, m, b, center, r):
        x_center=center[0]
        y_center=center[1]
        a = 1 + m**2
        b1 = 2 * (m * (b - y_center) - x_center)
        c = x_center**2 + (b - y_center)**2 - r**2
        discriminant = b1**2 - 4 * a * c

        if discriminant < 0:
            
            return None
        else:
            x1 = (-b1 + math.sqrt(discriminant)) / (2 * a)
            x2 = (-b1 - math.sqrt(discriminant)) / (2 * a)

            y1 = m * x1 + b
            y2 = m * x2 + b

            intersection_points = [(x1, y1), (x2, y2)]
            return intersection_points


    def is_point_between(self, p1, p2, p3):
        distance_p1_p2 = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        distance_p1_p3 = math.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
        distance_p2_p3 = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
        return math.isclose(distance_p1_p3 + distance_p2_p3, distance_p1_p2)


    def agents_point(self, goal_pole1, goal_pole2, center, radius):
        #print(goal_pole1, goal_pole2, center, radius)
        mid=[(goal_pole1[0]+goal_pole2[0])/2, (goal_pole1[1]+goal_pole2[1])/2]
        m, c=self.find_slope(mid, center)
        points=self.find_intersection(m, c, center, radius)
        return points[1] if self.is_point_between(mid, center, points[0]) else points[0]
    
    def generate_coordinates(self, i, random_start=True):
        # if i<20:
        #     start_agent_x = 0
        #     start_agent_y = 0
        #     target_x = 3
        #     target_y = 3
        #     return start_agent_x, start_agent_y, target_x, target_y
        if random_start==True:
            start_agent_x = random.uniform(-4.3, 4.3)
            start_agent_y = random.uniform(-4.3, 4.3)
        else:
            start_agent_x = 0
            start_agent_y = 0
        target_x = random.uniform(-4.3, 4.3)
        target_y = random.uniform(-4.3, 4.3)
        #target_x, target_y = #agents_point(goal_pole1, goal_pole2, center, 0.5)
        return start_agent_x, start_agent_y, target_x, target_y
    def find_relative_target_coordinates(self, agent, target, ball):
        relative_target_x = target[0] - agent[0]
        relative_target_y = target[1] - agent[1]
        relative_ball_x = ball[0] - agent[0]
        relative_ball_y = ball[1] - agent[1]
        return relative_target_x, relative_target_y, relative_ball_x, relative_ball_y
    def find_action(self, state):
       #action = self.policy_net(state).argmax(dim=1)
       state=state.squeeze(0).numpy()
       angle_rad = math.atan2(state[3], state[2])
       angle_deg = math.degrees(angle_rad)
       angle_deg = (angle_deg + 180) % 360 - 180
       return angle_rad
class contacts:
    def Is_agent_in_ground(agentcoord, ballcoord):
        if -5 <= agentcoord[0] <= 5 and -5 <= agentcoord[1] <= 5:
            return True
        print("out of ground")
        return False
    def Is_ball_touched(relative_ball_x, relative_ball_y):
        return np.sqrt(relative_ball_x**2 + relative_ball_y**2)<0.2
    def Is_goal(relative_target_x, relative_target_y):
        return np.sqrt(relative_target_x**2 + relative_target_y**2)<0.2
class rewards:
    def __init__(self):
        self.helper_obj=helper()
    def cal_reward(self, relative_target_x, relative_target_y, relative_ball_x, relative_ball_y, agentcoord, targetcoord, ballcoord):
        next_relative_target_x, next_relative_target_y, next_relative_ball_x, next_relative_ball_y=self.helper_obj.find_relative_target_coordinates(agentcoord, targetcoord, ballcoord)
        if contacts.Is_goal(relative_target_x, relative_target_y)==True:
            print("Reached")
            return True, True, 100, next_relative_target_x, next_relative_target_y, next_relative_ball_x, next_relative_ball_y
        if contacts.Is_agent_in_ground(agentcoord, ballcoord)==False:
            return False, True, -100000, next_relative_target_x, next_relative_target_y, next_relative_ball_x, next_relative_ball_y
        if contacts.Is_ball_touched(relative_ball_x, relative_ball_y)==True:
            print("ball touched")
            return False, True, -100, next_relative_target_x, next_relative_target_y, next_relative_ball_x, next_relative_ball_y
        else:
            initial_distance = np.sqrt(relative_target_x**2 + relative_target_y**2)
            new_distance = np.sqrt(next_relative_target_x**2 + next_relative_target_y**2)
            delta_distance = initial_distance - new_distance
            if np.sqrt(relative_ball_x**2 + relative_ball_y**2)<0.3:
                return False, False, -delta_distance, next_relative_target_x, next_relative_target_y, next_relative_ball_x, next_relative_ball_y
            return False, False, delta_distance, next_relative_target_x, next_relative_target_y, next_relative_ball_x, next_relative_ball_y

