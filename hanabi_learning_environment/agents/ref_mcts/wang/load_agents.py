# Developed by Lorenzo Mambretti, Justin Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://github.com/jtwwang/hanabi/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
from hanabi_learning_environment import agents

def load_agent(agent_class):
    """
    import the module required for a specific class and returns
    the agent correspondent to that class
    """
    # MB: REduced to the ones known
    if agent_class == 'MCAgent':
        from hanabi_learning_environment.agents.ref_mcts.wang.wang_mcts import MCAgent
        agent = MCAgent
    elif agent_class == 'RandomAgent':
        from hanabi_learning_environment.agents.random_agent import RandomAgent
    elif agent_class == 'SimpleAgent':
        from hanabi_learning_environment.agents.simple_agent import SimpleAgent
        agent = SimpleAgent
    else:
        raise ValueError("Invalid agent_class %s" %agent_class)

    return agent
