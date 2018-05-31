import agent
import GUI
import MCTS


agent1 = MCTS.MCTS('C:\\Users\\nik\\Desktop\\server\\topmodel228',
                   'C:\\Users\\nik\\Desktop\\server\\topmodel228',
                   sim_number=10000,
                   linear=False,
                   time=14,
                   rollout_depth=10)
agent2 = agent.HumanAgent()
GUI.run_gui(agent1, agent2, delay=0)
