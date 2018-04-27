import agent
import GUI

agent1 = agent.HumanAgent()
agent2 = agent.SLAgent('C:\\Users\\nik\\Desktop\\server\\model1.4')
GUI.run_gui(agent1, agent2, delay=0)

