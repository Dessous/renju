import agent
import GUI
import MCTS

agent2 = agent.SLAgent('C:\\Users\\nik\\Documents\\GitHub\\renju\\weights\\model1.6')
agent1 = agent.HumanAgent()
agent2 = MCTS.MCTS('C:\\Users\\nik\\Desktop\\server\\light4',
                   'C:\\Users\\nik\\Desktop\\server\\rollout4',
                   sim_number=10000,
                   rollout_depth=10)
GUI.run_gui(agent1, agent2, delay=0)
