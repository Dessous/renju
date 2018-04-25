import renju
import agent

agent1 = agent.HumanAgent()
agent2 = agent.SLAgent('/home/dessous/renju/weights/model1.3')

print(renju.run_test(agent1, agent2))
