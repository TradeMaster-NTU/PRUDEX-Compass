import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import pandas as pd

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            log_dir,
            trainer,
            exploration_env,
            evaluation_env,
            test_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            test_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            save_frequency=1,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            test_env,
            exploration_data_collector,
            evaluation_data_collector,
            test_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.save_frequency = save_frequency
        self.log_dir = log_dir
        
    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        eval_tr, test_tr = [], []
        eval_sr, test_sr = [], []
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):

            print("Epoch:{}".format(epoch))
            print('eval!!!!!!!!!!!!!!')
            print(self.max_path_length)
            print(self.num_eval_steps_per_epoch),
            discard_incomplete_paths=True,
            eval_paths = self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            # print("eval_path",eval_paths)
            #eval_paths is [] when data=="sz50"
            eval_tr.append(eval_paths[0]['env_infos'][-1][0])
            eval_sr.append(eval_paths[0]['env_infos'][-1][1])
            print('test!!!!!')
            test_paths = self.test_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            test_tr.append(test_paths[0]['env_infos'][-1][0])
            test_sr.append(test_paths[0]['env_infos'][-1][1])
            daily_return = test_paths[0]['env_infos'][-1][2]
            daily_action=test_paths[0]['actions']

            path = self.log_dir + '/test_daily_return_{}.csv'.format(epoch)
            action_path = self.log_dir + '/test_daily_action_{}.npy'.format(epoch)
            daily_return.to_csv(path)
            import numpy as np
            np.save(action_path,daily_action)
            # gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)
            # don't need to print rl related statistics
            #self._end_epoch(epoch)
            if self.save_frequency > 0:
                if epoch % self.save_frequency == 0:
                    self.trainer.save_models(epoch)
                    self.replay_buffer.save_buffer(epoch)
        print('eval sharpe:{}'.format(eval_sr))
        print('test sharpe:{}'.format(test_sr))
        import pandas as pd
        df = pd.DataFrame(data={"eval_tr": eval_tr, "eval_sr": eval_sr, "test_tr": test_tr, "test_sr": test_sr})
        df.to_csv(self.log_dir + '/result.csv')