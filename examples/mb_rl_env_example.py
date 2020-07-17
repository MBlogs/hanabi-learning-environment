# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple episode runner using the RL environment."""

from __future__ import print_function
import sys
import getopt
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from hanabi_learning_environment.agents.mcts.mcts_agent import MCTSAgent
from hanabi_learning_environment.agents.mcts.ris_mcts_agent import RISMCTSAgent
# from hanabi_learning_environment.agents.ref_mcts.wang.wang_mcts import MCAgent

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'MCTSAgent': MCTSAgent}

class Runner(object):
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = {'players': flags['players']}
    self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
    self.agent_classes = [AGENT_CLASSES[agent_class] for agent_class in flags['agent_classes']]

  def run(self):
    """Run episodes."""
    rewards = []
    for episode in range(flags['num_episodes']):
      observations = self.environment.reset()
      # MB: Allow parsing of different Agents. N
      # TODO: Assumes config is the same for all agents. Will need to be different for config.
      agents = [agent_class(self.agent_config) for agent_class in self.agent_classes]
      done = False
      episode_reward = 0
      while not done:
        for agent_id, agent in enumerate(agents):
          observation = observations['player_observations'][agent_id]
          action = agent.act(observation)
          if observation['current_player'] == agent_id:
            assert action is not None
            current_player_action = action
          else:
            assert action is None
        # Make an environment step.
        print('Agent: {} action: {}'.format(observation['current_player'],
                                            current_player_action))

        observations, reward, done, unused_info = self.environment.step(current_player_action, True)
        episode_reward += reward

        # MB: Try a return and DealSpecifc Move upfront for the next player
        #return_action = {'action_type': 'RETURN', 'card_index': 0}
        #observations, reward, done, unused_info = self.environment.step(return_action)
        #print("After RETURN:")
        #print_state(self)

      # MB: Rewards seems pretty funky. It's zero for all non-perfect games? A: Yes may want to change that
      rewards.append(episode_reward)
      print('Running episode: %d' % episode)
      print('Max Reward: %.3f' % max(rewards))
    return rewards

def fireworks_score(fireworks):
  '''Utility function to return score'''
  score = 0
  for f, s in fireworks.items():
    score += s
  return score

def print_state(self):
  print("-----------------------------------------------\n{}".format(self.environment.state))

def print_observation(observation):
  ''' MB: Utility function. Print important information about the state'''
  # TODO: An observation in the rl_env is lacking the full card knowledge from the state
  print("\n")
  print('------- Observation from Player:{} -------'.format(observation['current_player']))
  print("Observation keys: {}".format(observation.keys()))
  print("Number of players: {}".format(observation['num_players']))
  print("Current Player: {}".format(observation['current_player']))
  print("Current Player Offset: {}".format(observation['current_player_offset']))
  print("Information Tokens: {}".format(observation['information_tokens']))
  print("Life Tokens {}".format(observation['life_tokens']))
  print("Deck size: {}".format(observation['deck_size']))
  print("Discard Pile: {}".format(observation['discard_pile']))
  print("Fireworks: {}".format(observation['fireworks']))
  print("Fireworks Score: {}".format(fireworks_score(observation['fireworks'])))
  print("Legal Moves: {}".format(observation['legal_moves']))
  print("Observed Hands: {}".format(observation['observed_hands']))
  print("Card Knowledge: {}".format(observation['card_knowledge']))
  print("Pyhanabi is the direct HanabiObservation class and has additional functions like")
  print(".card_knowledge() (list of HanabiCardKnowledge objects):{}".format(observation['pyhanabi'].card_knowledge()))
  print(".observed_hands() {}".format(observation['pyhanabi'].observed_hands()))
  print(".card_playable_on_fireworks(card,value)".format(observation['pyhanabi']))
  print("\n")


if __name__ == "__main__":
  # MB: agent_class changed to agent_classes
  flags = {'players': 3, 'num_episodes': 1, 'agent_classes': ['SimpleAgent', 'SimpleAgent', 'SimpleAgent']}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class='])
  if arguments:
    sys.exit('usage: rl_env_example.py [options]\n'
             '--players       number of players in the game.\n'
             '--num_episodes  number of game episodes to run.\n'
             '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
  # MB: Added check that the number of agent classes is equal to number of players
  if len(flags['agent_classes']) != flags['players']:
    sys.exit('Number of agent classes not same as number of players')

  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)
  runner = Runner(flags)
  runner.run()
