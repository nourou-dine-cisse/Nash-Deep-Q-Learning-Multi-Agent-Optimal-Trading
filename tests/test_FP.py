import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import *

env = TradingEnv()
state = env.reset()
done = False

advantage_net, value_net = AdvantageNet(), ValueNet()
print(fictitious_play(env, advantage_net, value_net, B=10))