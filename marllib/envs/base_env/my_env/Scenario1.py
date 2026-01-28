import numpy as np
from utils import BaseScenario, Agent, UWBPlanningWorld


class Scenario(BaseScenario):
    def make_world(self, agent_num=3):
        world = UWBPlanningWorld(map_size=50.0, map_resolution=0.5)
        # set any world properties first
        num_agents = agent_num
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.2
            agent.perceived_grid_map = np.ones_like(world.grid_map, dtype=np.int8)
        return world

    def reset_world(self, world, np_random):
        # 1) reset each agent's perceived map (belief) to unknown
        for agent in world.agents:
            agent.perceived_grid_map = np.ones_like(world.grid_map, dtype=np.int8)

        # 2) reset agent positions within half map size (map_size_m is full length)
        half = int(world.map_size_m / 2.0)

        for agent in world.agents:

            while True:
                agent_rx = np_random.randint(-half, half + 1)
                agent_ry = np_random.randint(-half, half + 1)

                agent_gx, agent_gy = world.to_grid(agent_rx, agent_ry)

                # ensure inside map bounds and not inside obstacles
                if (0 <= agent_gx < world.grid_map.shape[0] and 0 <= agent_gy < world.grid_map.shape[1]
                        and world.grid_map[agent_gx, agent_gy] == 0):
                    agent.state.p_pos = np.array([agent_rx, agent_ry], dtype=np.float32)
                    agent.state.p_vel = np.zeros(world.dim_p)
                    break

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def global_reward(self, world):
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
