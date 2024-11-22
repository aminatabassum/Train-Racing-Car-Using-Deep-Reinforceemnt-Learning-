from racetrack_env.env.racetrack import RaceTrack
from racetrack_env.algorithms.reinforce import REINFORCEAgent, REINFORCEAgentTrainer


if __name__ == "__main__":
    params = {
        'episode_num': 500,
        'learning_rate': 1e-2,
        'gamma': 0.99,
        'epsilon': 1e-5,
        'eye_sight': 2,
        'model_type': "numerical"
    }

    env = RaceTrack(
        track_file_path="tracks/simple.txt", 
        starting_position=(1, 2), 
        goal_positions=[
            (1, 6), (2, 6), (3, 6),
            (1, 5), (2, 5), (3, 5),
        ],
        eye_sight=params['eye_sight'],
        model_type=params['model_type'],
    )

    # print(env.reset()[0].shape)

    agent = REINFORCEAgent(params=params, model_type=params['model_type'])
    trainer = REINFORCEAgentTrainer(agent=agent, env=env, params=params)

    returns, losses = trainer.train()