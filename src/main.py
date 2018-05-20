import agent
import GUI
import MCTS

agent1=MCTS.MCTS('C:\\Users\\nik\\Documents\\GitHub\\renju\\weights\\model1.3',
                 'C:\\Users\\nik\\Documents\\GitHub\\renju\\weights\\model1.5',
                 sim_number=10)
agent2 = agent.HumanAgent()
GUI.run_gui(agent1, agent2, delay=0)
