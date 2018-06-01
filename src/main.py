import agent
import GUI
import MCTS

print('input side: \'black\' or \'white\'')
side = input()
print('input time limit for tree move (one number in seconds)')
move_time = float(input())
agent1 = MCTS.MCTS('../weights/gomoku_nn',
                   '../weights/gomoku_nn',
                   sim_number=10000,
                   linear=False,
                   time=move_time,
                   rollout_depth=10)
agent2 = agent.HumanAgent()
if side == 'black':
    agent1, agent2 = agent2, agent1
GUI.run_gui(agent1, agent2, delay=0)
