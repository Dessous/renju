import agent
import GUI
import MCTS


agent2 = agent.HumanAgent()
agent1 = MCTS.MCTS('C:\\Users\\nik\\Desktop\\server\\final2',
                   'C:\\Users\\nik\\Desktop\\server\\rollout4',
                   sim_number=10000,
                   rollout_depth=10)
GUI.run_gui(agent1, agent2, delay=0)
