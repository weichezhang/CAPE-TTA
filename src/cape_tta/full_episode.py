"""Full-horizon first-action interventions with injectable policy/environment."""


def roll_episode(env_factory, initial_state, first_action, policy, *, max_steps=220, settle_steps=10, on_step=None):
    if max_steps < 1 or settle_steps < 0:
        raise ValueError('Invalid rollout budget')
    env = env_factory()
    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        for _ in range(settle_steps):
            obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                raise RuntimeError('Task terminated during settling; intervention comparison invalid')
        for step in range(max_steps):
            action = list(first_action) if step == 0 else list(policy(obs))
            obs, _, done, _ = env.step(action)
            if on_step is not None:
                on_step({'step': step + 1, 'action': action, 'done': bool(done)})
            if done:
                return {'success': True, 'steps_executed': step + 1, 'termination': 'env_done'}
        return {'success': False, 'steps_executed': max_steps, 'termination': 'step_budget'}
    finally:
        env.close()
