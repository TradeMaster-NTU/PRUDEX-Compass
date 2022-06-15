import argparse
import rlkit.torch.pytorch_util as ptu
import random
import numpy as np
import torch
from rlkit.data_management.env_replay_buffer import EnsembleEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger_custom, set_seed
from rlkit.samplers.data_collector import EnsembleMdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.neurips20_sac_ensemble import NeurIPS20SACEnsembleTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rl_env.portfolio_management import TrainEnv,get_dataset_config

def parse_args():
    parser = argparse.ArgumentParser()
    # architecture
    parser.add_argument('--num_layer', default=2, type=int)
    
    # train
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--save_freq', default=0, type=int)

    # misc
    parser.add_argument('--seed', default=1, type=int)
    
    # env
    parser.add_argument('--env', default="portfolio_management", type=str)
    
    # ensemble
    parser.add_argument('--num_ensemble', default=3, type=int)
    parser.add_argument('--ber_mean', default=0.5, type=float)
    
    # inference
    parser.add_argument('--inference_type', default=0.0, type=float)
    
    # corrective feedback
    parser.add_argument('--temperature', default=20.0, type=float)

    # dataset
    parser.add_argument('--dataset', default='sz50', type=str)

    #
    parser.add_argument('--policy_lr', default=7E-4, type=float)
    parser.add_argument('--qf_lr', default=7E-4, type=float)
    parser.add_argument('--discount',default=0.99, type=float)
    parser.add_argument('--buffer_size',default=1E4,type=float)
    parser.add_argument('--layer_size', default=128, type=int)
    parser.add_argument('--uncertainty', default=0.5, type=float)

    args = parser.parse_args()
    return args

def get_env(env_name, config):
    if env_name in ['portfolio_management']:
        env = TrainEnv(config=config)
    return env

def set_seed(seed):
    """
    Set the seed for all the possible random number generators.

    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def experiment(variant):
    set_seed(variant['seed'])
    train_config, valid_config, test_config = get_dataset_config(variant['dataset'])
    expl_env = get_env(variant['env'], train_config)
    expl_env.seed(variant['seed'])
    eval_env = get_env(variant['env'], valid_config)
    eval_env.seed(variant['seed'])
    test_env = get_env(variant['env'], test_config)
    test_env.seed(variant['seed'])
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    
    M = variant['layer_size']
    num_layer = variant['num_layer']
    network_structure = [M] * num_layer
    
    NUM_ENSEMBLE = variant['num_ensemble']
    L_qf1, L_qf2, L_target_qf1, L_target_qf2, L_policy, L_eval_policy = [], [], [], [], [], []
    
    for _ in range(NUM_ENSEMBLE):
    
        qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        target_qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        target_qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=network_structure,
        )
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=network_structure,
        )
        eval_policy = MakeDeterministic(policy)
        
        L_qf1.append(qf1)
        L_qf2.append(qf2)
        L_target_qf1.append(target_qf1)
        L_target_qf2.append(target_qf2)
        L_policy.append(policy)
        L_eval_policy.append(eval_policy)
    
    eval_path_collector = EnsembleMdpPathCollector(
        eval_env,
        L_eval_policy,
        NUM_ENSEMBLE,
        eval_flag=True,
    )

    test_path_collector = EnsembleMdpPathCollector(
        test_env,
        L_eval_policy,
        NUM_ENSEMBLE,
        eval_flag=True,
    )
    
    expl_path_collector = EnsembleMdpPathCollector(
        expl_env,
        L_policy,
        NUM_ENSEMBLE,
        ber_mean=variant['ber_mean'],
        eval_flag=False,
        critic1=L_qf1,
        critic2=L_qf2,
        inference_type=variant['inference_type'],
        feedback_type=1,
    )

    replay_buffer = EnsembleEnvReplayBuffer(
        variant['buffer_size'],
        expl_env,
        NUM_ENSEMBLE,
        log_dir=variant['log_dir'],
    )
    
    trainer = NeurIPS20SACEnsembleTrainer(
        env=eval_env,
        policy=L_policy,
        qf1=L_qf1,
        qf2=L_qf2,
        target_qf1=L_target_qf1,
        target_qf2=L_target_qf2,
        num_ensemble=NUM_ENSEMBLE,
        feedback_type=1,
        temperature=variant['temperature'],
        temperature_act=0,
        expl_gamma=0,
        log_dir=variant['log_dir'],
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        log_dir= variant['log_dir'],
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        test_env=test_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        test_data_collector=test_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    args = parse_args()
    
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=args.layer_size,
        buffer_size=int(args.buffer_size),
        algorithm_kwargs=dict(
            num_epochs=10,
            num_eval_steps_per_epoch=1500,
            # num_eval_steps_per_epoch=300,
            # sz_50
            num_trains_per_train_loop=1500,
            num_expl_steps_per_train_loop=1500,
            min_num_steps_before_training=10000,
            # dj_30
            # num_trains_per_train_loop=2000,
            # num_expl_steps_per_train_loop=2000,
            # min_num_steps_before_training=70000,
            # acl_18
            # num_trains_per_train_loop=1500,
            # num_expl_steps_per_train_loop=1500,
            # min_num_steps_before_training=20000,
            max_path_length=5000,
            batch_size=args.batch_size,
            save_frequency=args.save_freq,
        ),
        trainer_kwargs=dict(
            discount=args.discount,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        num_ensemble=args.num_ensemble,
        num_layer=args.num_layer,
        dataset=args.dataset,
        seed=args.seed,
        ber_mean=args.ber_mean,
        env=args.env,
        inference_type=args.inference_type,
        temperature=args.temperature,
        log_dir="",
        uncertainty=args.uncertainty,

    )

    exp_name = 'AlphaMix+'

    ptu.set_gpu_mode(True)
    #for seed in [51234]:
    for seed in [12345,23451,34512,45123,51234]:
        variant['seed']=seed
        print('seed:{}------------'.format(seed))
        log_dir = setup_logger_custom(exp_name, variant=variant)
        variant['log_dir'] = log_dir
        ptu.set_gpu_mode(True)
        experiment(variant)

