import renju
import agent

agent1 = agent.RandomAgent()
agent2 = agent.RandomAgent()

print(renju.run_test(agent1, agent2))
