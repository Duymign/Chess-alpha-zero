import os
class EvaluateConfig:
    def __init__(self):
        self.game_num = 10  # số lượng game để đánh giá
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 100
        self.play_config.thinking_loop = 1
        self.play_config.c_puct = 1 
        self.play_config.tau_decay_rate = 0.6
        self.play_config.noise_eps = 0
        self.evaluate_latest_first = True
        self.max_game_length = 1000


class PlayDataConfig:
    def __init__(self):
        self.min_elo_policy = 500  # tải xuống trò chơi có chất lượng tốt hơn với elo >= 500
        self.max_elo_policy = 1800  # tải xuống trò chơi có chất lượng tốt hơn với elo <= 1800
        self.sl_nb_game_in_file = 250
        self.nb_game_in_file = 5  # Giảm giá trị này để tạo nhiều tệp hơn
        self.max_file_num = 150
        self.save_policy_of_tau_1 = True


class PlayConfig:
    def __init__(self):
        self.max_processes = 3
        self.search_threads = 16
        self.simulation_num_per_move = 100  # Giảm số lượng giả lập cho train nhanh
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 1.5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3
        self.tau_decay_rate = 0.98
        self.virtual_loss = 3
        self.resign_threshold = -0.8
        self.min_resign_turn = 5
        self.max_game_length = 1000
        self.share_mtcs_info_in_self_play = False
        self.reset_mtcs_info_per_game = 1


class TrainerConfig:
    def __init__(self):
        self.min_games_to_begin_learn = 10
        self.min_data_size_to_learn = 0
        self.cleaning_processes = 1
        self.vram_frac = 1.0
        self.batch_size = 256  # Batch size khi training
        self.epoch_to_checkpoint = 1
        self.dataset_size = 100000
        self.start_total_steps = 0
        self.save_model_steps = 20
        self.load_data_steps = 10
        self.loss_weights = [1.0, 1.0]
        self.max_checkpoints_to_keep = 3


class ModelConfig:
    def __init__(self):
        self.cnn_filter_num = 128
        self.cnn_first_filter_size = 5
        self.cnn_filter_size = 3
        self.res_layer_num = 5  # Giảm số lượng lớp để nhanh hơn
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.distributed = False
        self.input_depth = 18
        self.input_height = 8
        self.input_width = 8


class ResourceConfig:
    def __init__(self):
        import os
        self.project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.data_dir = os.path.join(self.project_dir, "data")
        self.model_dir = os.path.join(self.data_dir, "model")
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")
        self.sl_best_config_path = os.path.join(self.model_dir, "sl_best_config.json")
        self.sl_best_weight_path = os.path.join(self.model_dir, "sl_best_weight.h5")
        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_config_path = os.path.join(self.next_generation_model_dir, "model_config.json")
        self.next_generation_weight_path = os.path.join(self.next_generation_model_dir, "model_weight.h5")
        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"
        self.self_play_game_idx_file = os.path.join(self.data_dir, "self_play_game_idx")
        self.self_play_game_idx = 0
        self.self_play_first_file = os.path.join(self.data_dir, "self_play_first")
        self.self_play_first = 1
        self.sl_game_idx_file = os.path.join(self.data_dir, "sl_game_idx")
        self.sl_game_idx = 0
        self.sl_next_generation_game_idx = 0
        self.sl_play_data_dir = os.path.join(self.data_dir, "sl_play_data")
        self.sl_play_data_filename_tmpl = "play_%s.json"
        self.record_dir = os.path.join(self.project_dir, "records")
        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")
        self.tensorboard_log_dir = os.path.join(self.log_dir, "tensorboard")
        self.resource_lock_path = os.path.join(self.data_dir, "resource_context")
        self.self_play_mcts_dir = None

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.record_dir, self.log_dir,
                self.next_generation_model_dir, self.sl_play_data_dir, self.tensorboard_log_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class Config:
    def __init__(self, config_type="mini"):
        self.opts = Options()
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.play = PlayConfig()
        self.play_data = PlayDataConfig()
        self.trainer = TrainerConfig()
        self.eval = EvaluateConfig()
        self.labels = {
            'e': {'e1': 0, 'e2': 1, 'e3': 2, 'e4': 3, 'e5': 4, 'e6': 5, 'e7': 6, 'e8': 7},
            'd': {'d1': 8, 'd2': 9, 'd3': 10, 'd4': 11, 'd5': 12, 'd6': 13, 'd7': 14, 'd8': 15},
            'c': {'c1': 16, 'c2': 17, 'c3': 18, 'c4': 19, 'c5': 20, 'c6': 21, 'c7': 22, 'c8': 23},
            'b': {'b1': 24, 'b2': 25, 'b3': 26, 'b4': 27, 'b5': 28, 'b6': 29, 'b7': 30, 'b8': 31},
            'a': {'a1': 32, 'a2': 33, 'a3': 34, 'a4': 35, 'a5': 36, 'a6': 37, 'a7': 38, 'a8': 39},
            'f': {'f1': 40, 'f2': 41, 'f3': 42, 'f4': 43, 'f5': 44, 'f6': 45, 'f7': 46, 'f8': 47},
            'g': {'g1': 48, 'g2': 49, 'g3': 50, 'g4': 51, 'g5': 52, 'g6': 53, 'g7': 54, 'g8': 55},
            'h': {'h1': 56, 'h2': 57, 'h3': 58, 'h4': 59, 'h5': 60, 'h6': 61, 'h7': 62, 'h8': 63},
        }


class Options:
    def __init__(self):
        self.new = False
        self.light = False
        self.distributed = False